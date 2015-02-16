package ntm

import (
	"fmt"
	"log"
	"math"
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

	// Wtm1BiasV returns the bias values for the memory heads at time t-1.
	Wtm1BiasV() [][]*BetaSimilarity
	// Mtm1BiasV returns the bias values for the memory at time t-1.
	Mtm1BiasV() *WrittenMemory

	// Weights loops through all internal weights of a controller.
	// For each weight, Weights calls the callback with a unique tag and a pointer to the weight.
	Weights(f func(*Unit))
	WeightsVerbose(f func(string, *Unit))

	NumWeights() int
	NumHeads() int
	MemoryN() int
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

		h := m.Controller.Heads()[i]
		for j, u := range h.units {
			if math.IsNaN(u.Val) {
				panic(fmt.Sprintf("%d", j))
			}
		}

	}
	m.Circuit = NewCircuit(m.Controller.Heads(), old.Circuit.WM)
	return &m
}

func (m *NTM) Backward() {
	m.Circuit.Backward()
	m.Controller.Backward()
}

func forwardBackward(c Controller, in, out [][]float64) []*NTM {
	wtm1s := make([]*Refocus, c.NumHeads())
	reads := make([]*Read, c.NumHeads())
	cas := make([]*ContentAddressing, c.NumHeads())
	for i := range reads {
		cas[i] = NewContentAddressing(c.Wtm1BiasV()[i])
		wtm1s[i] = &Refocus{Top: make([]Unit, c.MemoryN())}
		for j := range wtm1s[i].Top {
			wtm1s[i].Top[j].Val = cas[i].Top[j].Val
		}
		reads[i] = NewRead(wtm1s[i], c.Mtm1BiasV())
	}
	machines := make([]*NTM, len(in))
	empty := &NTM{
		Controller: c,
		Circuit:    &Circuit{W: wtm1s, R: reads, WM: c.Mtm1BiasV()},
	}

	machines[0] = NewNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = NewNTM(machines[t-1], in[t])
	}
	c.Weights(func(u *Unit) { u.Grad = 0 })
	//okt := len(out) - ((len(out)-2) / 2)
	for t := len(in) - 1; t >= 0; t-- {
		m := machines[t]

		//if t >= okt {
		y := out[t]
		for i := 0; i < len(y); i++ {
			m.Controller.Y()[i].Grad = m.Controller.Y()[i].Val - y[i]
		}
		//}
		m.Backward()
	}

	for i := range reads {
		reads[i].Backward()
		for j := range reads[i].W.Top {
			cas[i].Top[j].Grad += reads[i].W.Top[j].Grad
		}
		cas[i].Backward()
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

func (s *SGDMomentum) Train(x, y [][]float64, alpha, mt float64) []*NTM {
	machines := forwardBackward(s.C, x, y)
	i := 0
	s.C.Weights(func(w *Unit) {
		d := -alpha*w.Grad + mt*s.PrevD[i]
		w.Val += d
		s.PrevD[i] = d
		i++
	})
	return machines
}

type RMSProp struct {
	C Controller
	N []float64
	G []float64
	D []float64
}

func NewRMSProp(c Controller) *RMSProp {
	r := RMSProp{
		C: c,
		N: make([]float64, c.NumWeights()),
		G: make([]float64, c.NumWeights()),
		D: make([]float64, c.NumWeights()),
	}
	return &r
}

func (r *RMSProp) Train(x, y [][]float64, a, b, c, d float64) []*NTM {
	machines := forwardBackward(r.C, x, y)
	i := 0
	r.C.Weights(func(w *Unit) {
		r.N[i] = a*r.N[i] + (1-a)*w.Grad*w.Grad
		r.G[i] = a*r.G[i] + (1-a)*w.Grad
		r.D[i] = b*r.D[i] - c*w.Grad/math.Sqrt(r.N[i]-r.G[i]*r.G[i]+d)
		w.Val += r.D[i]
		if math.IsNaN(w.Val) {
			log.Printf("%f %f %f %f", w.Grad, r.N[i], r.G[i], r.D[i])
			panic("")
		}
		i++
	})
	return machines
}
