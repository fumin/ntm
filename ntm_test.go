package ntm

import (
	"math"
	"math/rand"
	"testing"
)

func TestNTM(t *testing.T) {
	times := 3
	x := makeTensor2(times, 2)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i]); j++ {
			x[i][j] = rand.NormFloat64()
		}
	}
	y := makeTensor2(times, 2)
	for i := 0; i < len(y); i++ {
		for j := 0; j < len(y[i]); j++ {
			y[i][j] = rand.NormFloat64()
		}
	}
	n := 3
	m := 2
	h1Size := 3
	numHeads := 2
	w := NewControllerWs(len(x[0]), len(y[0]), h1Size, numHeads, m)
	randVal3(w.Wh1r)
	randVal2(w.Wh1x)
	randVal2(w.Wyh1)
	randVal3(w.Wuh1)
	forwardBackward(w, n, x, y)

	l := loss(w, n, x, y)
	checkWyh1(t, w, n, x, y, l)
	checkWuh1(t, w, n, x, y, l)
	checkWh1r(t, w, n, x, y, l)
	checkWh1x(t, w, n, x, y, l)
}

func loss(w *ControllerWs, memoryN int, in, out [][]float64) float64 {
	m := len(w.Wh1r[0][0])
	numHeads := len(w.Wh1r[0])
	h1Size := len(w.Wh1r)

	// Initialize memory as in the function forwardBackward
	mem := makeTensorUnit2(memoryN, m)
	for i := 0; i < len(mem); i++ {
		for j := 0; j < len(mem[i]); j++ {
			mem[i][j].Val = 1
		}
	}
	wtm1s := make([]*Refocus, numHeads)
	for i := 0; i < len(wtm1s); i++ {
		wtm1s[i] = &Refocus{Top: make([]Unit, memoryN)}
		wtm1s[i].Top[0].Val = 1
	}
	reads := makeTensor2(numHeads, m)
	for i := 0; i < len(reads); i++ {
		for j := 0; j < len(reads[i]); j++ {
			reads[i][j] = 1
		}
	}

	prediction := makeTensor2(len(out), len(out[0]))
	for t := 0; t < len(in); t++ {
		x := in[t]
		h1 := make([]float64, h1Size)
		for i := 0; i < len(h1); i++ {
			var v float64 = 0
			for j := 0; j < len(w.Wh1r[i]); j++ {
				for k := 0; k < len(w.Wh1r[i][j]); k++ {
					v += w.Wh1r[i][j][k].Val * reads[j][k]
				}
			}
			for j := 0; j < len(w.Wh1x[i]); j++ {
				v += w.Wh1x[i][j].Val * x[j]
			}
			h1[i] = sigmoid(v)
		}
		for i := 0; i < len(prediction[t]); i++ {
			var v float64 = 0
			for j := 0; j < len(w.Wyh1[i]); j++ {
				v += w.Wyh1[i][j].Val * h1[j]
			}
			prediction[t][i] = sigmoid(v)
		}
		heads := make([]*Head, numHeads)
		for i := 0; i < len(heads); i++ {
			heads[i] = NewHead(m)
			heads[i].Wtm1 = wtm1s[i]
			for j := 0; j < len(heads[i].units); j++ {
				for k := 0; k < len(w.Wuh1[i][j]); k++ {
					heads[i].units[j].Val += w.Wuh1[i][j][k].Val * h1[k]
				}
			}
		}
		wsFloat64, readsFloat64, memFloat64 := doAddressing(heads, mem)
		wtm1s = transformWSFloat64(wsFloat64)
		reads = readsFloat64
		mem = transformMemFloat64(memFloat64)
	}

	var llh float64 = 0 // log likelihood
	for t := 0; t < len(out); t++ {
		for i := 0; i < len(out[t]); i++ {
			p := prediction[t][i]
			y := out[t][i]
			llh += y*math.Log(p) + (1-y)*math.Log(1-p)
		}
	}
	return -llh
}

func checkWyh1(t *testing.T, w *ControllerWs, memoryN int, in, out [][]float64, lx float64) {
	for i := 0; i < len(w.Wyh1); i++ {
		for j := 0; j < len(w.Wyh1[i]); j++ {
			x := w.Wyh1[i][j].Val
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			w.Wyh1[i][j].Val = xph
			lxph := loss(w, memoryN, in, out)
			w.Wyh1[i][j].Val = x
			grad := (lxph - lx) / (xph - x)

			if math.IsNaN(grad) || math.Abs(grad-w.Wyh1[i][j].Grad) > 1e-5 {
				t.Fatalf("wrong Wyh1[%d][%d] gradient expected %f, got %f", i, j, grad, w.Wyh1[i][j].Grad)
			} else {
				t.Logf("OK Wyh1[%d][%d] gradient expected %f, got %f", i, j, grad, w.Wyh1[i][j].Grad)
			}
		}
	}
}

func checkWuh1(t *testing.T, w *ControllerWs, memoryN int, in, out [][]float64, lx float64) {
	for i := 0; i < len(w.Wuh1); i++ {
		for j := 0; j < len(w.Wuh1[i]); j++ {
			for k := 0; k < len(w.Wuh1[i][j]); k++ {
				x := w.Wuh1[i][j][k].Val
				h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
				xph := x + h
				w.Wuh1[i][j][k].Val = xph
				lxph := loss(w, memoryN, in, out)
				w.Wuh1[i][j][k].Val = x
				grad := (lxph - lx) / (xph - x)

				if math.IsNaN(grad) || math.Abs(grad-w.Wuh1[i][j][k].Grad) > 1e-5 {
					t.Fatalf("wrong Wuh1[%d][%d][%d] gradient expected %f, got %f", i, j, k, grad, w.Wuh1[i][j][k].Grad)
				} else {
					t.Logf("OK Wuh1[%d][%d][%d] gradient expected %f, got %f", i, j, k, grad, w.Wuh1[i][j][k].Grad)
				}
			}
		}
	}
}

func checkWh1r(t *testing.T, w *ControllerWs, memoryN int, in, out [][]float64, lx float64) {
	for i := 0; i < len(w.Wh1r); i++ {
		for j := 0; j < len(w.Wh1r[i]); j++ {
			for k := 0; k < len(w.Wh1r[i][j]); k++ {
				x := w.Wh1r[i][j][k].Val
				h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
				xph := x + h
				w.Wh1r[i][j][k].Val = xph
				lxph := loss(w, memoryN, in, out)
				w.Wh1r[i][j][k].Val = x
				grad := (lxph - lx) / (xph - x)

				if math.IsNaN(grad) || math.Abs(grad-w.Wh1r[i][j][k].Grad) > 1e-5 {
					t.Fatalf("wrong Wh1r[%d][%d][%d] gradient expected %f, got %f", i, j, k, grad, w.Wh1r[i][j][k].Grad)
				} else {
					t.Logf("OK Wh1r[%d][%d][%d] gradient expected %f, got %f", i, j, k, grad, w.Wh1r[i][j][k].Grad)
				}
			}
		}
	}
}

func checkWh1x(t *testing.T, w *ControllerWs, memoryN int, in, out [][]float64, lx float64) {
	for i := 0; i < len(w.Wh1x); i++ {
		for j := 0; j < len(w.Wh1x[i]); j++ {
			x := w.Wh1x[i][j].Val
			h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
			xph := x + h
			w.Wh1x[i][j].Val = xph
			lxph := loss(w, memoryN, in, out)
			w.Wh1x[i][j].Val = x
			grad := (lxph - lx) / (xph - x)

			if math.IsNaN(grad) || math.Abs(grad-w.Wh1x[i][j].Grad) > 1e-5 {
				t.Fatalf("wrong Wh1x[%d][%d] gradient expected %f, got %f", i, j, grad, w.Wh1x[i][j].Grad)
			} else {
				t.Logf("OK Wh1x[%d][%d] gradient expected %f, got %f", i, j, grad, w.Wh1x[i][j].Grad)
			}
		}
	}
}

func transformMemFloat64(memFloat64 [][]float64) [][]Unit {
	mem := makeTensorUnit2(len(memFloat64), len(memFloat64[0]))
	for i := 0; i < len(mem); i++ {
		for j := 0; j < len(mem[0]); j++ {
			mem[i][j].Val = memFloat64[i][j]
		}
	}
	return mem
}

func transformWSFloat64(wsFloat64 [][]float64) []*Refocus {
	wtm1s := make([]*Refocus, len(wsFloat64))
	for i := 0; i < len(wtm1s); i++ {
		wtm1s[i] = &Refocus{Top: make([]Unit, len(wsFloat64[i]))}
		for j := 0; j < len(wtm1s[i].Top); j++ {
			wtm1s[i].Top[j].Val = wsFloat64[i][j]
		}
	}
	return wtm1s
}
