/*
Package ntm implements the Neural Turing Machine architecture as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.

Using this package along its subpackages, the "copy", "repeatcopy" and "ngram" tasks mentioned in the paper were verified.
For each of these tasks, the successfully trained models are saved under the filenames "seedA_B",
where A is the number indicating the seed provided to rand.Seed in the training process, and B is the iteration number in which the trained weights converged.
*/
package ntm

import (
	"math"

	"github.com/gonum/blas/blas64"
)

// A Head is a read write head on a memory bank.
// It carriess information that is required to operate on a memory bank according to the NTM architecture.
type Head struct {
	vals  []float64
	grads []float64

	Wtm1 *refocus // the weights at time t-1
	M    int      // size of a row in the memory
}

// NewHead creates a new memory head.
func NewHead(m int) *Head {
	h := Head{
		M: m,
	}
	return &h
}

// EraseVector returns the erase vector of a memory head.
func (h *Head) EraseVal() []float64 {
	return h.vals[0:h.M]
}

func (h *Head) EraseGrad() []float64 {
	return h.grads[0:h.M]
}

// AddVector returns the add vector of a memory head.
func (h *Head) AddVal() []float64 {
	return h.vals[h.M : 2*h.M]
}

func (h *Head) AddGrad() []float64 {
	return h.grads[h.M : 2*h.M]
}

// K returns a head's key vector, which is the target data in the content addressing step.
func (h *Head) KVal() []float64 {
	return h.vals[2*h.M : 3*h.M]
}

func (h *Head) KGrad() []float64 {
	return h.grads[2*h.M : 3*h.M]
}

// Beta returns the key strength of a content addressing step.
func (h *Head) BetaVal() *float64 {
	return &h.vals[3*h.M]
}

func (h *Head) BetaGrad() *float64 {
	return &h.grads[3*h.M]
}

// G returns the degree in which we want to choose content-addressing over location-based-addressing.
func (h *Head) GVal() *float64 {
	return &h.vals[3*h.M+1]
}

func (h *Head) GGrad() *float64 {
	return &h.grads[3*h.M+1]
}

// S returns a value indicating how much the weightings are rotated in a location-based-addressing step.
func (h *Head) SVal() *float64 {
	return &h.vals[3*h.M+2]
}

func (h *Head) SGrad() *float64 {
	return &h.grads[3*h.M+2]
}

// Gamma returns the degree in which the addressing weights are sharpened.
func (h *Head) GammaVal() *float64 {
	return &h.vals[3*h.M+3]
}

func (h *Head) GammaGrad() *float64 {
	return &h.grads[3*h.M+3]
}

func headUnitsLen(m int) int {
	return 3*m + 4
}

// The Controller interface is implemented by NTM controller networks that wish to operate with memory banks in a NTM.
type Controller interface {
	// Heads returns the emitted memory heads.
	Heads() []*Head
	// YVal returns the values of the output of the Controller.
	YVal() []float64
	// YVal returns the gradients of the output of the Controller.
	YGrad() []float64

	// Forward creates a new Controller which shares the same internal weights,
	// and performs a forward pass whose results can be retrived by Heads and Y.
	Forward(reads []*memRead, x []float64) Controller
	// Backward performs a backward pass,
	// assuming the gradients on Heads and Y are already set.
	Backward()

	// Wtm1BiasVal returns the values of the bias of the previous weight.
	// The layout is |-- 1st head weights (size memoryN) --|-- 2nd head --|-- ... --|
	// The length of the returned slice is numHeads * memoryN.
	Wtm1BiasVal() []float64
	Wtm1BiasGrad() []float64

	// M1mt1BiasVal returns the values of the bias of the memory bank.
	// The returned matrix is in row major order.
	Mtm1BiasVal() []float64
	Mtm1BiasGrad() []float64

	// WeightsVal returns the values of all weights.
	WeightsVal() []float64
	// WeightsGrad returns the gradients of all weights.
	WeightsGrad() []float64
	// WeightsDesc returns the descriptions of a weight.
	WeightsDesc(i int) string

	// NumHeads returns the number of memory heads of a controller.
	NumHeads() int
	// MemoryN returns the number of vectors of the memory bank of a controller.
	MemoryN() int
	// MemoryM returns the size of a vector in the memory bank of a controller.
	MemoryM() int
}

// A NTM is a neural turing machine as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.
type NTM struct {
	Controller Controller
	memOp      *memOp
}

// NewNTM creates a new NTM.
func NewNTM(old *NTM, x []float64) *NTM {
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
func ForwardBackward(c Controller, in [][]float64, out DensityModel) []*NTM {
	weights := c.WeightsGrad()
	for i := range weights {
		weights[i] = 0
	}

	// Set the empty NTM's memory and head weights to their bias values.
	empty, reads, cas := makeEmptyNTM(c)
	machines := make([]*NTM, len(in))

	// Backpropagation through time.
	machines[0] = NewNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = NewNTM(machines[t-1], in[t])
	}
	for t := len(in) - 1; t >= 0; t-- {
		m := machines[t]
		out.Model(t, m.Controller.YVal(), m.Controller.YGrad())
		m.backward()
	}

	// Compute gradients for the bias values of the initial memory and weights.
	for i := range reads {
		reads[i].Backward()
		for j := range reads[i].W.TopGrad {
			cas[i].Top[j].Grad += reads[i].W.TopGrad[j]
		}
		cas[i].Backward()
	}

	// Copy gradients to the controller.
	cwtm1 := c.Wtm1BiasGrad()
	for i := range cas {
		for j, bs := range cas[i].Units {
			cwtm1[i*c.MemoryN()+j] = bs.Top.Grad
		}
	}

	return machines
}

// MakeEmptyNTM makes a NTM with its memory and head weights set to their bias values, based on the controller.
func MakeEmptyNTM(c Controller) *NTM {
	machine, _, _ := makeEmptyNTM(c)
	return machine
}

func makeEmptyNTM(c Controller) (*NTM, []*memRead, []*contentAddressing) {
	cwtm1 := c.Wtm1BiasVal()
	unws := make([][]*betaSimilarity, c.NumHeads())
	for i := range unws {
		unws[i] = make([]*betaSimilarity, c.MemoryN())
		for j := range unws[i] {
			v := cwtm1[i*c.MemoryN()+j]
			unws[i][j] = &betaSimilarity{Top: Unit{Val: v}}
		}
	}

	mtm1 := &writtenMemory{
		N:       c.MemoryN(),
		TopVal:  c.Mtm1BiasVal(),
		TopGrad: c.Mtm1BiasGrad(),
	}

	wtm1s := make([]*refocus, c.NumHeads())
	reads := make([]*memRead, c.NumHeads())
	cas := make([]*contentAddressing, c.NumHeads())
	for i := range reads {
		cas[i] = newContentAddressing(unws[i])
		wtm1s[i] = &refocus{
			TopVal:  make([]float64, c.MemoryN()),
			TopGrad: make([]float64, c.MemoryN()),
		}
		for j := range wtm1s[i].TopVal {
			wtm1s[i].TopVal[j] = cas[i].Top[j].Val
		}
		reads[i] = newMemRead(wtm1s[i], mtm1)
	}

	empty := &NTM{
		Controller: c,
		memOp:      &memOp{W: wtm1s, R: reads, WM: mtm1},
	}

	return empty, reads, cas
}

// Predictions returns the predictions of a NTM across time.
func Predictions(machines []*NTM) [][]float64 {
	pdts := make([][]float64, len(machines))
	for t := range pdts {
		pdts[t] = machines[t].Controller.YVal()
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
			hws[i][t] = make([]float64, len(m.memOp.W[i].TopVal))
			for j, w := range m.memOp.W[i].TopVal {
				hws[i][t][j] = w
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
		PrevD: make([]float64, len(c.WeightsVal())),
	}
	return &s
}

func (s *SGDMomentum) Train(x [][]float64, y DensityModel, alpha, mt float64) []*NTM {
	machines := ForwardBackward(s.C, x, y)

	weights := s.C.WeightsVal()
	for i, grad := range s.C.WeightsGrad() {
		d := -alpha*grad + mt*s.PrevD[i]
		weights[i] += d
		s.PrevD[i] = d
	}
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
		N: make([]float64, len(c.WeightsVal())),
		G: make([]float64, len(c.WeightsVal())),
		D: make([]float64, len(c.WeightsVal())),
	}
	return &r
}

func (r *RMSProp) Train(x [][]float64, y DensityModel, a, b, c, d float64) []*NTM {
	machines := ForwardBackward(r.C, x, y)
	r.update(a, b, c, d)
	return machines
}

func (r *RMSProp) update(a, b, c, d float64) {
	grad := blas64.Vector{Inc: 1, Data: r.C.WeightsGrad()}
	grad2 := blas64.Vector{Inc: 1, Data: make([]float64, len(grad.Data))}
	for i, w := range grad.Data {
		grad2.Data[i] = w * w
	}

	n := blas64.Vector{Inc: 1, Data: r.N}
	blas64.Scal(len(n.Data), a, n)
	blas64.Axpy(len(n.Data), 1-a, grad2, n)

	g := blas64.Vector{Inc: 1, Data: r.G}
	blas64.Scal(len(g.Data), a, g)
	blas64.Axpy(len(g.Data), 1-a, grad, g)

	rms := blas64.Vector{Inc: 1, Data: make([]float64, len(r.D))}
	for i, g := range r.G {
		rms.Data[i] = grad.Data[i] / math.Sqrt(r.N[i]-g*g+d)
	}
	rD := blas64.Vector{Inc: 1, Data: r.D}
	blas64.Scal(len(rD.Data), b, rD)
	blas64.Axpy(len(rD.Data), -c, rms, rD)

	val := blas64.Vector{Inc: 1, Data: r.C.WeightsVal()}
	blas64.Axpy(len(rD.Data), 1, rD, val)
}

// A DensityModel is a model of how the last layer of a network gets transformed into the final output.
type DensityModel interface {
	// Model sets the value and gradient of Units of the output layer.
	Model(t int, yHVal []float64, yHGrad []float64)

	// Loss is the loss definition of this model.
	Loss(output [][]float64) float64
}

// A LogisticModel models its outputs as logistic sigmoids.
type LogisticModel struct {
	// Y is the strength of the output unit at each time step.
	Y [][]float64
}

// Model sets the values and gradients of the output units.
func (m *LogisticModel) Model(t int, yHVal []float64, yHGrad []float64) {
	ys := m.Y[t]
	for i, yhv := range yHVal {
		newYhv := Sigmoid(yhv)
		yHVal[i] = newYhv
		yHGrad[i] = newYhv - ys[i]
	}
}

// Loss returns the cross entropy loss.
func (m *LogisticModel) Loss(output [][]float64) float64 {
	var l float64 = 0
	for t, yh := range output {
		for i := range yh {
			p := output[t][i]
			y := m.Y[t][i]
			l += y*math.Log(p) + (1-y)*math.Log(1-p)
		}
	}
	return -l
}

// A MultinomialModel models its outputs as following the multinomial distribution.
type MultinomialModel struct {
	// Y is the class of the output at each time step.
	Y []int
}

// Model sets the values and gradients of the output units.
func (m *MultinomialModel) Model(t int, yHVal []float64, yHGrad []float64) {
	var sum float64 = 0
	for i, yhv := range yHVal {
		v := math.Exp(yhv)
		yHVal[i] = v
		sum += v
	}

	k := m.Y[t]
	for i, yhv := range yHVal {
		newYhv := yhv / sum
		yHVal[i] = newYhv
		yHGrad[i] = newYhv - delta(i, k)
	}
}

func (m *MultinomialModel) Loss(output [][]float64) float64 {
	var l float64 = 0
	for t, yh := range output {
		l += math.Log(yh[m.Y[t]])
	}
	return -l
}
