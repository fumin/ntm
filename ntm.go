package ntm

import (
//"log"
//"math"
//"math/rand"
)

type Head struct {
	units []Unit
	Wtm1  *Refocus // the weights at time t-1
	M     int      // size of a row in the memory
}

func NewHead(m int) *Head {
	h := Head{
		units: make([]Unit, 3*m+4),
		M:     m,
	}
	return &h
}

func (h *Head) EraseVector() []Unit {
	return h.units[0:h.M]
}

func (h *Head) AddVector() []Unit {
	return h.units[h.M : 2*h.M]
}

func (h *Head) K() []Unit {
	return h.units[2*h.M : 3*h.M]
}

func (h *Head) Beta() *Unit {
	return &h.units[3*h.M]
}

func (h *Head) G() *Unit {
	return &h.units[3*h.M+1]
}

func (h *Head) S() *Unit {
	return &h.units[3*h.M+2]
}

func (h *Head) Gamma() *Unit {
	return &h.units[3*h.M+3]
}

type ControllerWs struct {
	Wh1r [][][]Unit // weights from reads r to 1st hidden layer h1
	Wh1x [][]Unit

	Wyh1 [][]Unit
	Wuh1 [][][]Unit
}

func NewControllerWs(xSize, ySize, h1Size, numHeads, m int) *ControllerWs {
	h := NewHead(m)
	headUnitsSize := len(h.units)

	w := ControllerWs{
		Wh1r: makeTensorUnit3(h1Size, numHeads, m),
		Wh1x: makeTensorUnit2(h1Size, xSize),
		Wyh1: makeTensorUnit2(ySize, h1Size),
		Wuh1: makeTensorUnit3(numHeads, headUnitsSize, h1Size),
	}
	return &w
}

func (w *ControllerWs) ClearGradients() {
	clearGrad3(w.Wh1r)
	clearGrad2(w.Wh1x)
	clearGrad2(w.Wyh1)
	clearGrad3(w.Wuh1)
}

type Controller struct {
	W     *ControllerWs
	Reads []*Read
	X     []float64

	H1 []Unit

	Y     []Unit
	Heads []*Head
}

type ControllerConf struct {
	YSize int
	M     int
}

func NewController(w *ControllerWs, reads []*Read, x []float64, conf ControllerConf) *Controller {
	c := Controller{
		W:     w,
		Reads: reads,
		X:     x,
		H1:    make([]Unit, len(w.Wh1r)),
		Y:     make([]Unit, conf.YSize),
		Heads: make([]*Head, len(reads)),
	}

	for i := 0; i < len(c.H1); i++ {
		var v float64 = 0
		for j := 0; j < len(reads); j++ {
			for k := 0; k < len(reads[j].Top); k++ {
				v += w.Wh1r[i][j][k].Val * reads[j].Top[k].Val
			}
		}
		for j := 0; j < len(x); j++ {
			v += w.Wh1x[i][j].Val * x[j]
		}
		c.H1[i].Val = sigmoid(v)
	}

	for i := 0; i < len(c.Y); i++ {
		var v float64 = 0
		for j := 0; j < len(c.H1); j++ {
			v += w.Wyh1[i][j].Val * c.H1[j].Val
		}
		c.Y[i].Val = sigmoid(v)
	}
	for i := 0; i < len(c.Heads); i++ {
		c.Heads[i] = NewHead(conf.M)
		for j := 0; j < len(c.Heads[i].units); j++ {
			for k := 0; k < len(w.Wuh1[i][j]); k++ {
				c.Heads[i].units[j].Val += w.Wuh1[i][j][k].Val * c.H1[k].Val
			}
		}
	}

	return &c
}

func (c *Controller) Backward() {
	for i := 0; i < len(c.H1); i++ {
		var grad float64 = 0
		for j := 0; j < len(c.Y); j++ {
			grad += c.Y[j].Grad * c.W.Wyh1[j][i].Val
		}
		for j := 0; j < len(c.Heads); j++ {
			for k := 0; k < len(c.Heads[j].units); k++ {
				grad += c.Heads[j].units[k].Grad * c.W.Wuh1[j][k][i].Val
			}
		}
		c.H1[i].Grad += grad
	}
	for i := 0; i < len(c.W.Wyh1); i++ {
		for j := 0; j < len(c.W.Wyh1[i]); j++ {
			c.W.Wyh1[i][j].Grad += c.Y[i].Grad * c.H1[j].Val
		}
	}
	for i := 0; i < len(c.W.Wuh1); i++ {
		for j := 0; j < len(c.W.Wuh1[i]); j++ {
			for k := 0; k < len(c.W.Wuh1[i][j]); k++ {
				c.W.Wuh1[i][j][k].Grad += c.Heads[i].units[j].Grad * c.H1[k].Val
			}
		}
	}

	for i := 0; i < len(c.Reads); i++ {
		for j := 0; j < len(c.Reads[i].Top); j++ {
			for k := 0; k < len(c.H1); k++ {
				c.Reads[i].Top[j].Grad += c.H1[k].Grad * c.H1[k].Val * (1 - c.H1[k].Val) * c.W.Wh1r[k][i][j].Val
			}
		}
	}
	for i := 0; i < len(c.W.Wh1r); i++ {
		for j := 0; j < len(c.W.Wh1r[i]); j++ {
			for k := 0; k < len(c.W.Wh1r[i][j]); k++ {
				c.W.Wh1r[i][j][k].Grad += c.H1[i].Grad * c.H1[i].Val * (1 - c.H1[i].Val) * c.Reads[j].Top[k].Val
			}
		}
	}
	for i := 0; i < len(c.W.Wh1x); i++ {
		for j := 0; j < len(c.W.Wh1x[i]); j++ {
			c.W.Wh1x[i][j].Grad += c.H1[i].Grad * c.H1[i].Val * (1 - c.H1[i].Val) * c.X[j]
		}
	}
}

type NTM struct {
	Controller *Controller
	Circuit    *Circuit
}

func NewNTM(old *NTM, x []float64) *NTM {
	conf := ControllerConf{YSize: len(old.Controller.W.Wyh1), M: len(old.Circuit.WM.Top[0])}
	m := NTM{
		Controller: NewController(old.Controller.W, old.Circuit.R, x, conf),
	}
	for i := 0; i < len(m.Controller.Heads); i++ {
		m.Controller.Heads[i].Wtm1 = old.Circuit.W[i]
	}
	m.Circuit = NewCircuit(m.Controller.Heads, old.Circuit.WM)
	return &m
}

func (m *NTM) Backward(y []float64) {
	m.Circuit.Backward()
	for i := 0; i < len(y); i++ {
		m.Controller.Y[i].Grad = m.Controller.Y[i].Val - y[i]
	}
	m.Controller.Backward()
}

func forwardBackward(w *ControllerWs, memoryN int, in, out [][]float64) {
	numHeads := len(w.Wuh1)
	n := memoryN
	m := len(w.Wh1r[0][0])
	// Initialize all memories to 1. We cannot initialize to zero since we will
	// get a divide by zero in the cosine similarity of content addressing.
	refocuses := make([]*Refocus, numHeads)
	for i := 0; i < len(refocuses); i++ {
		refocuses[i] = &Refocus{Top: make([]Unit, n)}
		refocuses[i].Top[0].Val = 1
	}
	reads := make([]*Read, numHeads)
	for i := 0; i < len(reads); i++ {
		reads[i] = &Read{Top: make([]Unit, m)}
		for j := 0; j < len(reads[i].Top); j++ {
			reads[i].Top[j].Val = 1
		}
	}
	mem := &WrittenMemory{Top: makeTensorUnit2(n, m)}
	for i := 0; i < len(mem.Top); i++ {
		for j := 0; j < len(mem.Top[i]); j++ {
			mem.Top[i][j].Val = 1
		}
	}

	machines := make([]*NTM, len(in))
	empty := &NTM{
		Controller: &Controller{W: w},
		Circuit:    &Circuit{W: refocuses, R: reads, WM: mem},
	}
	machines[0] = NewNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = NewNTM(machines[t-1], in[t])
	}
	w.ClearGradients()
	for t := len(in) - 1; t >= 0; t-- {
		machines[t].Backward(out[t])
	}
}
