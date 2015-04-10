package ntm

import (
	"math"
	"math/rand"
	"testing"
)

func TestLogisticModel(t *testing.T) {
	times := 9
	x := makeTensor2(times, 4)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i]); j++ {
			x[i][j] = rand.Float64()
		}
	}
	y := makeTensor2(times, 4)
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
	weights := c.WeightsVal()
	for i := range weights {
		weights[i] = 2 * rand.Float64()
	}

	model := &LogisticModel{Y: y}
	ForwardBackward(c, x, model)
	checkGradients(t, c, Controller1Forward, x, model)
}

func TestMultinomialModel(t *testing.T) {
	times := 9
	x := makeTensor2(times, 4)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i]); j++ {
			x[i][j] = rand.Float64()
		}
	}
	outputSize := 4
	y := make([]int, times)
	for i := range y {
		y[i] = rand.Intn(outputSize)
	}
	n := 3
	m := 2
	h1Size := 3
	numHeads := 2
	c := NewEmptyController1(len(x[0]), outputSize, h1Size, numHeads, n, m)
	weights := c.WeightsVal()
	for i := range weights {
		weights[i] = 2 * rand.Float64()
	}

	model := &MultinomialModel{Y: y}
	ForwardBackward(c, x, model)
	checkGradients(t, c, Controller1Forward, x, model)
}

// A ControllerForward is a ground truth implementation of the forward pass of a controller.
type ControllerForward func(c Controller, reads [][]float64, x []float64) (prediction []float64, heads []*Head)

func Controller1Forward(c1 Controller, reads [][]float64, x []float64) ([]float64, []*Head) {
	c := c1.(*controller1)
	readX := make([]float64, 0)
	for _, read := range reads {
		for _, r := range read {
			readX = append(readX, r)
		}
	}
	for _, xi := range x {
		readX = append(readX, xi)
	}
	readX = append(readX, 1)
	h1 := make([]float64, c.h1Size)
	wh1 := c.wh1Val()
	for i := range h1 {
		var v float64 = 0
		for j, rx := range readX {
			v += wh1.Data[i*wh1.Cols+j] * rx
		}
		h1[i] = Sigmoid(v)
	}

	out := make([]float64, c.wyRows())
	wy := c.wyVal()
	h1 = append(h1, 1)
	for i := range out {
		var v float64 = 0
		for j, h := range h1 {
			v += wy.Data[i*wy.Cols+j] * h
		}
		out[i] = v
	}
	prediction := make([]float64, c.ySize)
	for i := range prediction {
		prediction[i] = out[i]
	}
	heads := make([]*Head, c.numHeads)
	for i := range heads {
		heads[i] = NewHead(c.memoryM)
		hul := headUnitsLen(c.MemoryM())
		heads[i].vals = make([]float64, hul)
		heads[i].grads = make([]float64, hul)
		for j := range heads[i].vals {
			heads[i].vals[j] += out[c.ySize+i*hul+j]
		}
	}

	return prediction, heads
}

func loss(c Controller, forward ControllerForward, in [][]float64, model DensityModel) float64 {
	// Initialize memory as in the function ForwardBackward
	mem := makeTensorUnit2(c.MemoryN(), c.MemoryM())
	for i := range mem {
		for j := range mem[i] {
			mem[i][j].Val = c.Mtm1BiasVal()[i*c.MemoryM()+j]
		}
	}
	wtm1s := make([]*refocus, c.NumHeads())
	for i := range wtm1s {
		wtm1s[i] = &refocus{
			TopVal:  make([]float64, c.MemoryN()),
			TopGrad: make([]float64, c.MemoryN()),
		}
		bs := c.Wtm1BiasVal()[i*c.MemoryN() : (i+1)*c.MemoryN()]
		var sum float64 = 0
		for j, b := range bs {
			wtm1s[i].TopVal[j] = math.Exp(b)
			sum += wtm1s[i].TopVal[j]
		}
		for j := range bs {
			wtm1s[i].TopVal[j] = wtm1s[i].TopVal[j] / sum
		}
	}
	reads := makeTensor2(c.NumHeads(), c.MemoryM())
	for i := 0; i < len(reads); i++ {
		for j := 0; j < len(reads[i]); j++ {
			var v float64 = 0
			for k := 0; k < len(mem); k++ {
				v += wtm1s[i].TopVal[k] * mem[k][j].Val
			}
			reads[i][j] = v
		}
	}

	prediction := make([][]float64, len(in))
	var heads []*Head
	for t := 0; t < len(in); t++ {
		prediction[t], heads = forward(c, reads, in[t])
		prediction[t] = computeDensity(t, prediction[t], model)
		for i := 0; i < len(heads); i++ {
			heads[i].Wtm1 = wtm1s[i]
		}
		wsFloat64, readsFloat64, memFloat64 := doAddressing(heads, mem)
		wtm1s = transformWSFloat64(wsFloat64)
		reads = readsFloat64
		mem = transformMemFloat64(memFloat64)
	}

	return model.Loss(prediction)
}

func computeDensity(timestep int, pred []float64, model DensityModel) []float64 {
	den := make([]float64, len(pred))
	copy(den, pred)
	model.Model(timestep, den, make([]float64, len(pred)))
	return den
}

func checkGradients(t *testing.T, c Controller, forward ControllerForward, in [][]float64, model DensityModel) {
	lx := loss(c, forward, in, model)

	for i, x := range c.WeightsVal() {
		h := machineEpsilonSqrt * math.Max(math.Abs(x), 1)
		xph := x + h
		c.WeightsVal()[i] = xph
		lxph := loss(c, forward, in, model)
		c.WeightsVal()[i] = x
		grad := (lxph - lx) / (xph - x)

		wGrad := c.WeightsGrad()[i]
		tag := c.WeightsDesc(i)
		if math.IsNaN(grad) || math.Abs(grad-wGrad) > 1e-5 {
			t.Errorf("wrong %s gradient expected %f, got %f", tag, grad, wGrad)
		} else {
			t.Logf("OK %s gradient expected %f, got %f", tag, grad, wGrad)
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

func transformWSFloat64(wsFloat64 [][]float64) []*refocus {
	wtm1s := make([]*refocus, len(wsFloat64))
	for i := 0; i < len(wtm1s); i++ {
		wtm1s[i] = &refocus{
			TopVal:  make([]float64, len(wsFloat64[i])),
			TopGrad: make([]float64, len(wsFloat64[i])),
		}
		for j := 0; j < len(wtm1s[i].TopVal); j++ {
			wtm1s[i].TopVal[j] = wsFloat64[i][j]
		}
	}
	return wtm1s
}
