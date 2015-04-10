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
	memory := &writtenMemory{
		N:       n,
		TopVal:  make([]float64, n*m),
		TopGrad: make([]float64, n*m),
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			memory.TopVal[i*m+j] = rand.Float64()
		}
	}
	hul := headUnitsLen(m)
	heads := make([]*Head, 2)
	for i := 0; i < len(heads); i++ {
		heads[i] = NewHead(m)
		heads[i].vals = make([]float64, hul)
		heads[i].grads = make([]float64, hul)
		heads[i].Wtm1 = randomRefocus(n)
		for j := 0; j < hul; j++ {
			heads[i].vals[j] = rand.Float64()
		}
	}
	// We want to check the case where Beta > 0 and Gamma > 1.
	*heads[0].BetaVal() = 0.137350
	*heads[0].GammaVal() = 1.9876

	circuit := newMemOp(heads, memory)
	for i := 0; i < len(circuit.W); i++ {
		for j := 0; j < len(circuit.W[i].TopGrad); j++ {
			if i == 0 && j == 0 {
				circuit.W[i].TopGrad[j] += w11OutputGradient
			} else {
				circuit.W[i].TopGrad[j] += outputGradient
			}
		}
	}
	for i := 0; i < len(circuit.R); i++ {
		for j := 0; j < len(circuit.R[i].TopGrad); j++ {
			circuit.R[i].TopGrad[j] += outputGradient
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			circuit.WM.TopGrad[i*m+j] += outputGradient
		}
	}
	circuit.Backward()

	memoryTop := makeTensorUnit2(n, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			memoryTop[i][j].Val = memory.TopVal[i*m+j]
			memoryTop[i][j].Grad = memory.TopGrad[i*m+j]
		}
	}
	ax := addressing(heads, memoryTop)
	checkGamma(t, heads, memoryTop, ax)
	checkS(t, heads, memoryTop, ax)
	checkG(t, heads, memoryTop, ax)
	checkWtm1(t, heads, memoryTop, ax)
	checkBeta(t, heads, memoryTop, ax)
	checkK(t, heads, memoryTop, ax)
	checkMemory(t, heads, memoryTop, ax)
}

func addressing(heads []*Head, memory [][]Unit) float64 {
	return addressingLoss(doAddressing(heads, memory))
}

func doAddressing(heads []*Head, memory [][]Unit) (weights [][]float64, reads [][]float64, newMem [][]float64) {
	weights = makeTensor2(len(heads), len(memory))
	for i, h := range heads {
		// Content-based addressing
		beta := math.Exp(*h.BetaVal())
		wc := make([]float64, len(memory))
		var sum float64 = 0
		for j := 0; j < len(wc); j++ {
			wc[j] = math.Exp(beta * cosineSimilarity(h.KVal(), unitVals(memory[j])))
			sum += wc[j]
		}
		for j := 0; j < len(wc); j++ {
			wc[j] = wc[j] / sum
		}

		// Content-based, location-based addressing gate
		g := Sigmoid(*h.GVal())
		for j := 0; j < len(wc); j++ {
			wc[j] = g*wc[j] + (1-g)*h.Wtm1.TopVal[j]
		}

		// Location-based addressing
		n := len(weights[i])
		//s := math.Mod(h.S().Val, float64(n))
		//if s < 0 {
		//	s += float64(n)
		//}
		s := math.Mod((2*Sigmoid(*h.SVal())-1)+float64(n), float64(n))
		for j := 0; j < n; j++ {
			imj := (j + int(s)) % n
			simj := 1 - (s - math.Floor(s))
			weights[i][j] = wc[imj]*simj + wc[(imj+1)%n]*(1-simj)
		}

		// Refocusing
		gamma := math.Log(math.Exp(*h.GammaVal())+1) + 1
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
		eraseVec := heads[k].EraseVal()
		for i := 0; i < len(erase[k]); i++ {
			erase[k][i] = Sigmoid(eraseVec[i])
		}
		addVec := heads[k].AddVal()
		for i := 0; i < len(add[k]); i++ {
			add[k][i] = Sigmoid(addVec[i])
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
		for i := 0; i < len(hd.KVal()); i++ {
			x := hd.KVal()[i]
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			hd.KVal()[i] = xph
			dx := xph - x
			axph := addressing(heads, memory)
			grad := (axph - ax) / dx
			hd.KVal()[i] = x

			if math.IsNaN(grad) || math.Abs(grad-hd.KGrad()[i]) > 1e-5 {
				t.Fatalf("wrong beta[%d] gradient expected %f, got %f", i, grad, hd.KGrad()[i])
			} else {
				t.Logf("OK K[%d][%d] agradient %f %f", k, i, grad, hd.KGrad()[i])
			}
		}
	}
}

func checkBeta(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := *hd.BetaVal()
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		*hd.BetaVal() = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		*hd.BetaVal() = x

		if math.IsNaN(grad) || math.Abs(grad-(*hd.BetaGrad())) > 1e-5 {
			t.Fatalf("wrong beta gradient expected %f, got %f", grad, *hd.BetaGrad())
		} else {
			t.Logf("OK beta[%d] agradient %f %f", k, grad, *hd.BetaGrad())
		}
	}
}

func checkWtm1(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		for i := 0; i < len(hd.Wtm1.TopVal); i++ {
			x := hd.Wtm1.TopVal[i]
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			hd.Wtm1.TopVal[i] = xph
			dx := xph - x
			axph := addressing(heads, memory)
			grad := (axph - ax) / dx
			hd.Wtm1.TopVal[i] = x

			if math.IsNaN(grad) || math.Abs(grad-hd.Wtm1.TopGrad[i]) > 1e-5 {
				t.Fatalf("wrong wtm1[%d] gradient expected %f, got %f", i, grad, hd.Wtm1.TopGrad[i])
			} else {
				t.Logf("OK wtm1[%d][%d] agradient %f %f", k, i, grad, hd.Wtm1.TopGrad[i])
			}
		}
	}
}

func checkG(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := *hd.GVal()
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		*hd.GVal() = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		*hd.GVal() = x

		if math.IsNaN(grad) || math.Abs(grad-(*hd.GGrad())) > 1e-5 {
			t.Fatalf("wrong G gradient expected %f, got %f", grad, *hd.GGrad())
		} else {
			t.Logf("OK G[%d] agradient %f %f", k, grad, *hd.GGrad())
		}
	}
}

func checkS(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := *hd.SVal()
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		*hd.SVal() = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		*hd.SVal() = x

		if math.IsNaN(grad) || math.Abs(grad-(*hd.SGrad())) > 1e-5 {
			t.Fatalf("wrong S gradient expected %f, got %f", grad, *hd.SGrad())
		} else {
			t.Logf("OK S[%d] agradient %f %f", k, grad, *hd.SGrad())
		}
	}
}

func checkGamma(t *testing.T, heads []*Head, memory [][]Unit, ax float64) {
	for k, hd := range heads {
		x := *hd.GammaVal()
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		*hd.GammaVal() = xph
		dx := xph - x
		axph := addressing(heads, memory)
		grad := (axph - ax) / dx
		*hd.GammaVal() = x

		if math.IsNaN(grad) || math.Abs(grad-(*hd.GammaGrad())) > 1e-5 {
			t.Fatalf("wrong gamma gradient expected %f, got %f", grad, *hd.GammaGrad())
		} else {
			t.Logf("OK gamma[%d] gradient %f %f", k, grad, *hd.GammaGrad())
		}
	}
}

func randomRefocus(n int) *refocus {
	w := make([]float64, n)
	var sum float64 = 0
	for i := 0; i < len(w); i++ {
		w[i] = math.Abs(rand.Float64())
		sum += w[i]
	}
	for i := 0; i < len(w); i++ {
		w[i] = w[i] / sum
	}
	return &refocus{
		TopVal:  w,
		TopGrad: make([]float64, n),
	}
}

func unitVals(units []Unit) []float64 {
	v := make([]float64, 0, len(units))
	for _, u := range units {
		v = append(v, u.Val)
	}
	return v
}
