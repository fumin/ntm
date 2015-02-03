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

type Controller interface {
	// Heads returns the emitted memory heads.
	Heads() []*Head
	// Y returns the output of the Controller.
	Y() []Unit

	// Forward creates a new Controller which shares the same internal weights,
	// and performs a forward pass whose results can be retrived by Heads and Y.
	Forward(reads []*Read, x []float64) Controller
	// Backward performs a backward pass,
	// assuming the gradients on Heads and Y are already set.
	Backward()

	// Weights returns a channel which emits the internal weights of a controller.
	// Callers are expected to range the returning channel until it is closed to avoid
	// leaking goroutines.
	Weights() chan *Unit

	// ClearGradients sets the gradients of all internal weights of a controller to zero.
	ClearGradients()

	NumWeights() int
	NumHeads() int
	MemoryM() int
}

type NTM struct {
	Controller Controller
	Circuit    *Circuit
}

func NewNTM(old *NTM, x []float64) *NTM {
	m := NTM{
		Controller: old.Controller.Forward(old.Circuit.R, x),
	}
	for i := 0; i < len(m.Controller.Heads()); i++ {
		m.Controller.Heads()[i].Wtm1 = old.Circuit.W[i]
	}
	m.Circuit = NewCircuit(m.Controller.Heads(), old.Circuit.WM)
	return &m
}

func (m *NTM) Backward(y []float64) {
	m.Circuit.Backward()
	for i := 0; i < len(y); i++ {
		m.Controller.Y()[i].Grad = m.Controller.Y()[i].Val - y[i]
	}
	m.Controller.Backward()
}

func forwardBackward(c Controller, memoryN int, in, out [][]float64) []*NTM {
	n := memoryN
	m := c.MemoryM()
	// Initialize all memories to 1. We cannot initialize to zero since we will
	// get a divide by zero in the cosine similarity of content addressing.
	refocuses := make([]*Refocus, c.NumHeads())
	for i := 0; i < len(refocuses); i++ {
		refocuses[i] = &Refocus{Top: make([]Unit, n)}
		refocuses[i].Top[0].Val = 1
	}
	reads := make([]*Read, c.NumHeads())
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
		Controller: c,
		Circuit:    &Circuit{W: refocuses, R: reads, WM: mem},
	}
	machines[0] = NewNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = NewNTM(machines[t-1], in[t])
	}
	c.ClearGradients()
	for t := len(in) - 1; t >= 0; t-- {
		machines[t].Backward(out[t])
	}
	return machines
}

type SGDMomentum struct {
	C     Controller
	PrevD []float64
}

func NewSGDMomentum(c Controller) *SGDMomentum {
	s := SGDMomentum{
		C:     c,
		PrevD: make([]float64, c.NumWeights()),
	}
	return &s
}

func (s *SGDMomentum) Train(x, y [][]float64, n int, alpha, mt float64) []*NTM {
	machines := forwardBackward(s.C, n, x, y)
	i := 0
	for w := range s.C.Weights() {
		d := -alpha*w.Grad + mt*s.PrevD[i]
		w.Val += d
		s.PrevD[i] = d
		i++
	}
	return machines
}
