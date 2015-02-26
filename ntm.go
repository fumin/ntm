/*
Package ntm implements the Neural Turing Machine architecture as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.

Using this package along its subpackages, the "copy", "repeatcopy" and "ngram" tasks mentioned in the paper were verified.
For each of these tasks, the successfully trained models are saved under the filenames "seedA_B",
where A is the number indicating the seed provided to rand.Seed in the training process, and B is the iteration number in which the trained weights converged.
*/
package ntm

import (
	"math"
)

// A Head is a read write head on a memory bank.
// It carriess every information that is required to operate on a memory bank according to the NTM architecture..
type Head struct {
	units []Unit
	Wtm1  *refocus // the weights at time t-1
	M     int      // size of a row in the memory
}

// NewHead creates a new memory head.
func NewHead(m int) *Head {
	h := Head{
		units: make([]Unit, 3*m+4),
		M:     m,
	}
	return &h
}

// EraseVector returns the erase vector of a memory head.
func (h *Head) EraseVector() []Unit {
	return h.units[0:h.M]
}

// AddVector returns the add vector of a memory head.
func (h *Head) AddVector() []Unit {
	return h.units[h.M : 2*h.M]
}

// K returns a head's key vector, which is the target data in the content addressing step.
func (h *Head) K() []Unit {
	return h.units[2*h.M : 3*h.M]
}

// Beta returns the key strength of a content addressing step.
func (h *Head) Beta() *Unit {
	return &h.units[3*h.M]
}

// G returns the degree in which we want to choose content-addressing over location-based-addressing.
func (h *Head) G() *Unit {
	return &h.units[3*h.M+1]
}

// S returns a value indicating how much the weightings are rotated in a location-based-addressing step.
func (h *Head) S() *Unit {
	return &h.units[3*h.M+2]
}

// Gamma returns the degree in which the addressing weights are sharpened.
func (h *Head) Gamma() *Unit {
	return &h.units[3*h.M+3]
}

// The Controller interface is implemented by NTM controller networks that wish to operate with memory banks in a NTM.
type Controller interface {
	// Heads returns the emitted memory heads.
	Heads() []*Head
	// Y returns the output of the Controller.
	Y() []Unit

	// Forward creates a new Controller which shares the same internal weights,
	// and performs a forward pass whose results can be retrived by Heads and Y.
	Forward(reads []*memRead, x []float64) Controller
	// Backward performs a backward pass,
	// assuming the gradients on Heads and Y are already set.
	Backward()

	// Wtm1BiasV returns the bias values for the memory heads at time t-1.
	Wtm1BiasV() [][]*betaSimilarity
	// Mtm1BiasV returns the bias values for the memory at time t-1.
	Mtm1BiasV() *writtenMemory

	// Weights loops through all internal weights of a controller.
	// For each weight, Weights calls the callback with a unique tag and a pointer to the weight.
	Weights(f func(*Unit))
	WeightsVerbose(f func(string, *Unit))

	NumWeights() int
	NumHeads() int
	MemoryN() int
	MemoryM() int
}

// A NTM is a neural turing machine as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.
type NTM struct {
	Controller Controller
	memOp      *memOp
}

func newNTM(old *NTM, x []float64) *NTM {
	m := NTM{
		Controller: old.Controller.Forward(old.memOp.R, x),
	}
	for i := 0; i < len(m.Controller.Heads()); i++ {
		m.Controller.Heads()[i].Wtm1 = old.memOp.W[i]
	}
	m.memOp = newMemOp(m.Controller.Heads(), old.memOp.WM)
	return &m
}

func (m *NTM) backward() {
	m.memOp.Backward()
	m.Controller.Backward()
}

// ForwardBackward computes a controller's prediction and gradients with respect to the given ground truth input and output values.
func ForwardBackward(c Controller, in, out [][]float64) []*NTM {
	// Set memory and head weights to their bias values.
	wtm1s := make([]*refocus, c.NumHeads())
	reads := make([]*memRead, c.NumHeads())
	cas := make([]*contentAddressing, c.NumHeads())
	for i := range reads {
		cas[i] = newContentAddressing(c.Wtm1BiasV()[i])
		wtm1s[i] = &refocus{Top: make([]Unit, c.MemoryN())}
		for j := range wtm1s[i].Top {
			wtm1s[i].Top[j].Val = cas[i].Top[j].Val
		}
		reads[i] = newMemRead(wtm1s[i], c.Mtm1BiasV())
	}
	machines := make([]*NTM, len(in))
	empty := &NTM{
		Controller: c,
		memOp:      &memOp{W: wtm1s, R: reads, WM: c.Mtm1BiasV()},
	}

	// Backpropagation through time.
	machines[0] = newNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = newNTM(machines[t-1], in[t])
	}
	c.Weights(func(u *Unit) { u.Grad = 0 })
	for t := len(in) - 1; t >= 0; t-- {
		m := machines[t]
		y := out[t]
		for i := 0; i < len(y); i++ {
			m.Controller.Y()[i].Grad = m.Controller.Y()[i].Val - y[i]
		}
		m.backward()
	}

	// Compute gradients for the bias values of the initial memory and weights.
	for i := range reads {
		reads[i].Backward()
		for j := range reads[i].W.Top {
			cas[i].Top[j].Grad += reads[i].W.Top[j].Grad
		}
		cas[i].Backward()
	}

	return machines
}

// Loss returns the cross-entropy loss of a NTM.
func Loss(output [][]float64, ms []*NTM) float64 {
	var l float64 = 0
	for t := 0; t < len(output); t++ {
		for i := 0; i < len(output[t]); i++ {
			y := output[t][i]
			p := ms[t].Controller.Y()[i].Val
			l += y*math.Log2(p) + (1-y)*math.Log2(1-p)
		}
	}
	return -l
}

// Predictions returns the predictions of a NTM across time.
func Predictions(machines []*NTM) [][]float64 {
	pdts := make([][]float64, len(machines))
	for t := range pdts {
		y := machines[t].Controller.Y()
		pdts[t] = make([]float64, len(y))
		for i, v := range y {
			pdts[t][i] = v.Val
		}
	}
	return pdts
}

// HeadWeights returns the addressing weights of all memory heads across time.
// The top level elements represent each head.
// The second level elements represent every time instant.
func HeadWeights(machines []*NTM) [][][]float64 {
	hws := make([][][]float64, len(machines[0].memOp.W))
	for i := range hws {
		hws[i] = make([][]float64, len(machines))
		for t, m := range machines {
			hws[i][t] = make([]float64, len(m.memOp.W[i].Top))
			for j, w := range m.memOp.W[i].Top {
				hws[i][t][j] = w.Val
			}
		}
	}
	return hws
}

// SGDMomentum implements stochastic gradient descent with momentum.
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
	machines := ForwardBackward(s.C, x, y)
	i := 0
	s.C.Weights(func(w *Unit) {
		d := -alpha*w.Grad + mt*s.PrevD[i]
		w.Val += d
		s.PrevD[i] = d
		i++
	})
	return machines
}

// RMSProp implements the rmsprop algorithm. The detailed updating equations are given in
// Graves, Alex (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.
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
	machines := ForwardBackward(r.C, x, y)
	i := 0
	r.C.Weights(func(w *Unit) {
		r.N[i] = a*r.N[i] + (1-a)*w.Grad*w.Grad
		r.G[i] = a*r.G[i] + (1-a)*w.Grad
		r.D[i] = b*r.D[i] - c*w.Grad/math.Sqrt(r.N[i]-r.G[i]*r.G[i]+d)
		w.Val += r.D[i]
		i++
	})
	return machines
}
