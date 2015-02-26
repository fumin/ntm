package ntm

import (
	"math"
	"math/rand"
	"testing"
)

func TestController1(t *testing.T) {
	times := 10
	x := MakeTensor2(times, 4)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i]); j++ {
			x[i][j] = rand.Float64()
		}
	}
	y := MakeTensor2(times, 4)
	for i := 0; i < len(y); i++ {
		for j := 0; j < len(y[i]); j++ {
			y[i][j] = rand.Float64()
		}
	}
	n := 3
	m := 2
	h1Size := 3
	numHeads := 2
	c := NewEmptyController1(len(x[0]), len(y[0]), h1Size, numHeads, n, m)
	c.Weights(func(u *Unit) { u.Val = 2 * rand.Float64() })
	ForwardBackward(c, x, y)

	l := loss(c, Controller1Forward, x, y)
	checkGradients(t, c, Controller1Forward, x, y, l)
}

func Controller1Forward(c1 Controller, reads [][]float64, x []float64) ([]float64, []*Head) {
	c := c1.(*controller1)
	h1Size := len(c.Wh1r)
	h1 := make([]float64, h1Size)
	for i := 0; i < len(h1); i++ {
		var v float64 = 0
		for j := 0; j < len(c.Wh1r[i]); j++ {
			for k := 0; k < len(c.Wh1r[i][j]); k++ {
				v += c.Wh1r[i][j][k].Val * reads[j][k]
			}
		}
		for j := 0; j < len(c.Wh1x[i]); j++ {
			v += c.Wh1x[i][j].Val * x[j]
		}
		v += c.Wh1b[i].Val
		h1[i] = Sigmoid(v)
	}
	prediction := make([]float64, len(c.Wyh1))
	for i := 0; i < len(prediction); i++ {
		var v float64 = 0
		maxJ := len(c.Wyh1[i]) - 1
		for j := 0; j < maxJ; j++ {
			v += c.Wyh1[i][j].Val * h1[j]
		}
		v += c.Wyh1[i][maxJ].Val
		prediction[i] = Sigmoid(v)
	}
	numHeads := len(c.Wh1r[0])
	m := len(c.Wh1r[0][0])
	heads := make([]*Head, numHeads)
	for i := 0; i < len(heads); i++ {
		heads[i] = NewHead(m)
		for j := 0; j < len(heads[i].units); j++ {
			maxK := len(c.Wuh1[i][j]) - 1
			for k := 0; k < maxK; k++ {
				heads[i].units[j].Val += c.Wuh1[i][j][k].Val * h1[k]
			}
			heads[i].units[j].Val += c.Wuh1[i][j][maxK].Val
		}
	}
	return prediction, heads
}

func loss(c Controller, forward func(Controller, [][]float64, []float64) ([]float64, []*Head), in, out [][]float64) float64 {
	// Initialize memory as in the function ForwardBackward
	mem := c.Mtm1BiasV().Top
	wtm1Bs := c.Wtm1BiasV()
	wtm1s := make([]*refocus, c.NumHeads())
	for i := range wtm1s {
		wtm1s[i] = &refocus{Top: make([]Unit, c.MemoryN())}
		var sum float64 = 0
		for j := range wtm1Bs[i] {
			wtm1s[i].Top[j].Val = math.Exp(wtm1Bs[i][j].Top.Val)
			sum += wtm1s[i].Top[j].Val
		}
		for j := range wtm1Bs[i] {
			wtm1s[i].Top[j].Val = wtm1s[i].Top[j].Val / sum
		}
	}
	reads := MakeTensor2(c.NumHeads(), c.MemoryM())
	for i := 0; i < len(reads); i++ {
		for j := 0; j < len(reads[i]); j++ {
			var v float64 = 0
			for k := 0; k < len(mem); k++ {
				v += wtm1s[i].Top[k].Val * mem[k][j].Val
			}
			reads[i][j] = v
		}
	}

	prediction := make([][]float64, len(out))
	var heads []*Head
	for t := 0; t < len(in); t++ {
		prediction[t], heads = forward(c, reads, in[t])
		for i := 0; i < len(heads); i++ {
			heads[i].Wtm1 = wtm1s[i]
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

func checkGradients(t *testing.T, c Controller, forward func(Controller, [][]float64, []float64) ([]float64, []*Head), in, out [][]float64, lx float64) {
	c.WeightsVerbose(func(tag string, w *Unit) {
		x := w.Val
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		w.Val = xph
		lxph := loss(c, forward, in, out)
		w.Val = x
		grad := (lxph - lx) / (xph - x)

		if math.IsNaN(grad) || math.Abs(grad-w.Grad) > 1e-5 {
			t.Errorf("wrong %s gradient expected %f, got %f", tag, grad, w.Grad)
		} else {
			t.Logf("OK %s gradient expected %f, got %f", tag, grad, w.Grad)
		}
	})
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

func transformWSFloat64(wsFloat64 [][]float64) []*refocus {
	wtm1s := make([]*refocus, len(wsFloat64))
	for i := 0; i < len(wtm1s); i++ {
		wtm1s[i] = &refocus{Top: make([]Unit, len(wsFloat64[i]))}
		for j := 0; j < len(wtm1s[i].Top); j++ {
			wtm1s[i].Top[j].Val = wsFloat64[i][j]
		}
	}
	return wtm1s
}
