package ntm

import (
	"math"
	"math/rand"
	"testing"
)

const (
	// outputGradient is the gradient of all units at the output of a Circuit,
	// except for the output weights[1][1].
	// The gradient of weights[1][1], gw11, needs to be different from the other weights[i][j],
	// because the value of the `addressing` function to be dependent on gw11.
	// If gw11 == gwij, then since weights[i][j] always sum up to 1, gwij has no effect on `addressing`.
	outputGradient    = 1.234
	w11OutputGradient = 0.987
)

func TestCircuit(t *testing.T) {
	n := 3
	m := 2
	memory := &WrittenMemory{Top: makeTensorUnit2(n, m)}
	for i := 0; i < len(memory.Top); i++ {
		for j := 0; j < len(memory.Top[i]); j++ {
			memory.Top[i][j].Val = rand.NormFloat64()
		}
	}
	heads := make([]*Head, 2)
	for i := 0; i < len(heads); i++ {
		heads[i] = NewHead(m)
		heads[i].Wtm1 = randomRefocus(n)
		for j := 0; j < len(heads[i].units); j++ {
			heads[i].units[j].Val = rand.NormFloat64()
		}
	}
	// We want to check the case where Beta > 0 and Gamma > 1.
	heads[0].Beta().Val = 0.137350
	heads[0].Gamma().Val = 1.9876

	circuit := NewCircuit(heads, memory)
	for i := 0; i < len(circuit.W); i++ {
		for j := 0; j < len(circuit.W[i].Top); j++ {
			if i == 0 && j == 0 {
				circuit.W[i].Top[j].Grad += w11OutputGradient
			} else {
				circuit.W[i].Top[j].Grad += outputGradient
			}
		}
	}
	for i := 0; i < len(circuit.R); i++ {
		for j := 0; j < len(circuit.R[i].Top); j++ {
			circuit.R[i].Top[j].Grad += outputGradient
		}
	}
	for i := 0; i < len(circuit.WM.Top); i++ {
		for j := 0; j < len(circuit.WM.Top[i]); j++ {
			circuit.WM.Top[i][j].Grad += outputGradient
		}
	}
	circuit.Backward()

	ax := addressing(heads, memory.Top)
	checkGamma(t, heads, memory.Top, ax)
	checkS(t, heads, memory.Top, ax)
	checkG(t, heads, memory.Top, ax)
	checkWtm1(t, heads, memory.Top, ax)
	checkBeta(t, heads, memory.Top, ax)
	checkK(t, heads, memory.Top, ax)
	checkMemory(t, heads, memory.Top, ax)
}

func addressing(heads []*Head, memory [][]Unit) float64 {
	return addressingLoss(doAddressing(heads, memory))
}

func doAddressing(heads []*Head, memory [][]Unit) (weights [][]float64, reads [][]float64, newMem [][]float64) {
	weights = makeTensor2(len(heads), len(memory))
	for i, h := range heads {
		// Content-based addressing
		beta := math.Max(h.Beta().Val, 0)
		wc := make([]float64, len(memory))
		var sum float64 = 0
		for j := 0; j < len(wc); j++ {
			wc[j] = math.Exp(beta * similarity(unitVals(h.K()), unitVals(memory[j])))
			sum += wc[j]
		}
		for j := 0; j < len(wc); j++ {
			wc[j] = wc[j] / sum
		}

		// Content-based, location-based addressing gate
		g := sigmoid(h.G().Val)
		for j := 0; j < len(wc); j++ {
			wc[j] = g*wc[j] + (1-g)*h.Wtm1.Top[j].Val
		}

		// Location-based addressing
		n := len(weights[i])
		s := math.Mod(h.S().Val, float64(n))
		if s < 0 {
			s += float64(n)
		}
		for j := 0; j < n; j++ {
			imj := (j + int(s)) % n
			simj := 1 - (s - math.Floor(s))
			weights[i][j] = wc[imj]*simj + wc[(imj+1)%n]*(1-simj)
		}

		// Refocusing
		gamma := math.Max(h.Gamma().Val, 1)
		sum = 0
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = math.Pow(weights[i][j], gamma)
			sum += weights[i][j]
		}
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = weights[i][j] / sum
		}
	}

	reads = makeTensor2(len(heads), len(memory[0]))
	for i, w := range weights {
		r := reads[i]
		for j := 0; j < len(r); j++ {
			for k := 0; k < len(w); k++ {
				r[j] += w[k] * memory[k][j].Val
			}
		}
	}

	erase := makeTensor2(len(heads), len(memory[0]))
	add := makeTensor2(len(heads), len(memory[0]))
	for k := 0; k < len(heads); k++ {
		eraseVec := heads[k].EraseVector()
		for i := 0; i < len(erase[k]); i++ {
			erase[k][i] = sigmoid(eraseVec[i].Val)
		}
		addVec := heads[k].AddVector()
		for i := 0; i < len(add[k]); i++ {
			add[k][i] = sigmoid(addVec[i].Val)
		}
	}
	newMem = makeTensor2(len(memory), len(memory[0]))
	for i := 0; i < len(newMem); i++ {
		for j := 0; j < len(newMem[i]); j++ {
			newMem[i][j] = memory[i][j].Val
			for k := 0; k < len(heads); k++ {
				newMem[i][j] = newMem[i][j] * (1 - weights[k][i]*erase[k][j])
			}
			for k := 0; k < len(heads); k++ {
				newMem[i][j] += weights[k][i] * add[k][j]
			}
		}
	}
	return weights, reads, newMem
}

func addressingLoss(weights [][]float64, reads [][]float64, newMem [][]float64) float64 {
	var res float64 = 0
	for i, w := range weights {
		for j, v := range w {
			if i == 0 && j == 0 {
				res += v * w11OutputGradient
			} else {
				res += v * outputGradient
			}
		}
	}
	for _, r := range reads {
		for _, rr := range r {
			res += rr * outputGradient
		}
	}
	for i := 0; i < len(newMem); i++ {
		for j := 0; j < len(newMem[i]); j++ {
			res += newMem[i][j] * outputGradient
		}
	}
	return res
}

func checkMemory(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for i := 0; i < len(memory); i++ {
		for j := 0; j < len(memory[i]); j++ {
			x := memory[i][j].Val
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			memory[i][j].Val = xph
			dx := xph - x
			axph := addressing(heads, memory)
			grad := (axph - ax) / dx
			memory[i][j].Val = x

			if math.IsNaN(grad) || math.Abs(grad-memory[i][j].Grad) > 1e-5 {
				t.Fatalf("wrong memory gradient expected %f, got %f", grad, memory[i][j].Grad)
			} else {
				t.Logf("OK memory[%d][%d] gradient %f, %f", i, j, grad, memory[i][j].Grad)
			}
		}
	}
}

func checkK(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		for i := 0; i < len(hd.K()); i++ {
			x := hd.K()[i].Val
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			hd.K()[i].Val = xph
			dx := xph - x
			axph := addressing(heads, memory)
			grad := (axph - ax) / dx
			hd.K()[i].Val = x

			if math.IsNaN(grad) || math.Abs(grad-hd.K()[i].Grad) > 1e-5 {
				t.Fatalf("wrong beta[%d] gradient expected %f, got %f", i, grad, hd.K()[i].Grad)
			} else {
				t.Logf("OK K[%d][%d] agradient %f %f", k, i, grad, hd.K()[i].Grad)
			}
		}
	}
}

func checkBeta(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := hd.Beta().Val
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		hd.Beta().Val = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		hd.Beta().Val = x

		if math.IsNaN(grad) || math.Abs(grad-hd.Beta().Grad) > 1e-5 {
			t.Fatalf("wrong beta gradient expected %f, got %f", grad, hd.Beta().Grad)
		} else {
			t.Logf("OK beta[%d] agradient %f %f", k, grad, hd.Beta().Grad)
		}
	}
}

func checkWtm1(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		for i := 0; i < len(hd.Wtm1.Top); i++ {
			x := hd.Wtm1.Top[i].Val
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			hd.Wtm1.Top[i].Val = xph
			dx := xph - x
			axph := addressing(heads, memory)
			grad := (axph - ax) / dx
			hd.Wtm1.Top[i].Val = x

			if math.IsNaN(grad) || math.Abs(grad-hd.Wtm1.Top[i].Grad) > 1e-5 {
				t.Fatalf("wrong wtm1[%d] gradient expected %f, got %f", i, grad, hd.Wtm1.Top[i].Grad)
			} else {
				t.Logf("OK wtm1[%d][%d] agradient %f %f", k, i, grad, hd.Wtm1.Top[i].Grad)
			}
		}
	}
}

func checkG(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := hd.G().Val
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		hd.G().Val = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		hd.G().Val = x

		if math.IsNaN(grad) || math.Abs(grad-hd.G().Grad) > 1e-5 {
			t.Fatalf("wrong G gradient expected %f, got %f", grad, hd.G().Grad)
		} else {
			t.Logf("OK G[%d] agradient %f %f", k, grad, hd.G().Grad)
		}
	}
}

func checkS(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := hd.S().Val
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		hd.S().Val = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		hd.S().Val = x

		if math.IsNaN(grad) || math.Abs(grad-hd.S().Grad) > 1e-5 {
			t.Fatalf("wrong S gradient expected %f, got %f", grad, hd.S().Grad)
		} else {
			t.Logf("OK S[%d] agradient %f %f", k, grad, hd.S().Grad)
		}
	}
}

func checkGamma(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := hd.Gamma().Val
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		hd.Gamma().Val = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		hd.Gamma().Val = x

		if math.IsNaN(grad) || math.Abs(grad-hd.Gamma().Grad) > 1e-5 {
			t.Fatalf("wrong gamma gradient expected %f, got %f", grad, hd.Gamma().Grad)
		} else {
			t.Logf("OK gamma[%d] gradient %f %f", k, grad, hd.Gamma().Grad)
		}
	}
}

func randomRefocus(n int) *Refocus {
	w := make([]Unit, n)
	var sum float64 = 0
	for i := 0; i < len(w); i++ {
		w[i].Val = math.Abs(rand.NormFloat64())
		sum += w[i].Val
	}
	for i := 0; i < len(w); i++ {
		w[i].Val = w[i].Val / sum
	}
	return &Refocus{Top: w}
}
