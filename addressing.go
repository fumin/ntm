package ntm

import (
	"log"
	"math"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
)

type similarityCircuit struct {
	UVal    []float64
	UGrad   []float64
	VVal    []float64
	VGrad   []float64
	TopVal  float64
	TopGrad float64

	UV    float64
	Unorm float64
	Vnorm float64
}

func newSimilarityCircuit(uVal, uGrad, vVal, vGrad []float64) *similarityCircuit {
	s := similarityCircuit{
		UVal:  uVal,
		UGrad: uGrad,
		VVal:  vVal,
		VGrad: vGrad,
	}
	u := blas64.Vector{Inc: 1, Data: uVal}
	v := blas64.Vector{Inc: 1, Data: vVal}
	s.UV = blas64.Dot(len(uVal), u, v)
	s.Unorm = blas64.Nrm2(len(uVal), u)
	s.Vnorm = blas64.Nrm2(len(vVal), v)
	s.TopVal = s.UV / (s.Unorm * s.Vnorm)
	return &s
}

func (s *similarityCircuit) Backward() {
	uvuu := s.UV / (s.Unorm * s.Unorm)
	uvvv := s.UV / (s.Vnorm * s.Vnorm)
	uvg := s.TopGrad / (s.Unorm * s.Vnorm)
	u := blas64.Vector{Inc: 1, Data: s.UVal}
	v := blas64.Vector{Inc: 1, Data: s.VVal}

	ugrad := blas64.Vector{Inc: 1, Data: s.UGrad}
	blas64.Axpy(len(s.UGrad), uvg, v, ugrad)
	blas64.Axpy(len(s.UGrad), -uvuu*uvg, u, ugrad)

	vgrad := blas64.Vector{Inc: 1, Data: s.VGrad}
	blas64.Axpy(len(s.VGrad), uvg, u, vgrad)
	blas64.Axpy(len(s.VGrad), -uvvv*uvg, v, vgrad)
}

type betaSimilarity struct {
	BetaVal  *float64
	BetaGrad *float64

	S   *similarityCircuit
	Top Unit

	b float64
}

func newBetaSimilarity(betaVal *float64, betaGrad *float64, s *similarityCircuit) *betaSimilarity {
	bs := betaSimilarity{
		BetaVal:  betaVal,
		BetaGrad: betaGrad,
		S:        s,
		b:        math.Exp(*betaVal), // Beta is in the range (-Inf, Inf)
	}
	bs.Top.Val = bs.b * s.TopVal
	return &bs
}

func (bs *betaSimilarity) Backward() {
	*bs.BetaGrad += bs.S.TopVal * bs.b * bs.Top.Grad
	bs.S.TopGrad += bs.b * bs.Top.Grad
}

type contentAddressing struct {
	Units []*betaSimilarity
	Top   []Unit
}

func newContentAddressing(units []*betaSimilarity) *contentAddressing {
	s := contentAddressing{
		Units: units,
		Top:   make([]Unit, len(units)),
	}
	// Increase numerical stability by subtracting all weights by their max,
	// before computing math.Exp().
	var max float64 = -math.MaxFloat64
	for _, u := range s.Units {
		max = math.Max(max, u.Top.Val)
	}
	var sum float64 = 0
	for i, u := range s.Units {
		w := math.Exp(u.Top.Val - max)
		s.Top[i].Val = w
		sum += w
	}
	for i, top := range s.Top {
		s.Top[i].Val = top.Val / sum
	}
	return &s
}

func (s *contentAddressing) Backward() {
	var gv float64 = 0
	for _, top := range s.Top {
		gv += top.Grad * top.Val
	}
	for i, top := range s.Top {
		s.Units[i].Top.Grad += (top.Grad - gv) * top.Val
	}
}

type gatedWeighting struct {
	GVal  *float64
	GGrad *float64

	WC   *contentAddressing
	Wtm1 *refocus // the weights at time t-1
	Top  []Unit
}

func newGatedWeighting(gVal *float64, gGrad *float64, wc *contentAddressing, wtm1 *refocus) *gatedWeighting {
	wg := gatedWeighting{
		GVal:  gVal,
		GGrad: gGrad,
		WC:    wc,
		Wtm1:  wtm1,
		Top:   make([]Unit, len(wc.Top)),
	}
	gt := Sigmoid(*gVal)
	for i := 0; i < len(wg.Top); i++ {
		wg.Top[i].Val = gt*wc.Top[i].Val + (1-gt)*wtm1.TopVal[i]
	}
	return &wg
}

func (wg *gatedWeighting) Backward() {
	gt := Sigmoid(*wg.GVal)

	var grad float64 = 0
	for i := 0; i < len(wg.Top); i++ {
		grad += (wg.WC.Top[i].Val - wg.Wtm1.TopVal[i]) * wg.Top[i].Grad
	}
	*wg.GGrad += grad * gt * (1 - gt)

	for i := 0; i < len(wg.WC.Top); i++ {
		wg.WC.Top[i].Grad += gt * wg.Top[i].Grad
	}

	for i := 0; i < len(wg.Wtm1.TopGrad); i++ {
		wg.Wtm1.TopGrad[i] += (1 - gt) * wg.Top[i].Grad
	}
}

type shiftedWeighting struct {
	SVal  *float64
	SGrad *float64

	Z   float64
	WG  *gatedWeighting
	Top []Unit
}

func newShiftedWeighting(sVal *float64, sGrad *float64, wg *gatedWeighting) *shiftedWeighting {
	sw := shiftedWeighting{
		SVal:  sVal,
		SGrad: sGrad,
		WG:    wg,
		Top:   make([]Unit, len(wg.Top)),
	}

	n := len(sw.WG.Top)
	//sw.Z = math.Mod(s.Val, float64(n))
	//if sw.Z < 0 {
	//	sw.Z += float64(n)
	//}

	//sw.Z = float64(n) * Sigmoid(s.Val)
	shift := (2*Sigmoid(*sVal) - 1) // * maxShift
	sw.Z = math.Mod(shift+float64(n), float64(n))

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		sw.Top[i].Val = sw.WG.Top[imj].Val*simj + sw.WG.Top[(imj+1)%n].Val*(1-simj)
		if math.IsNaN(sw.Top[i].Val) || sw.Top[i].Val < 0 {
			log.Printf("imj: %d, wg: %f, simj: %f, wg+1: %f", imj, sw.WG.Top[imj].Val, simj, sw.WG.Top[(imj+1)%n].Val)
			panic("")
		}
	}
	return &sw
}

func (sw *shiftedWeighting) Backward() {
	var grad float64 = 0
	n := len(sw.WG.Top)
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		grad += (-sw.WG.Top[imj].Val + sw.WG.Top[(imj+1)%n].Val) * sw.Top[i].Grad
	}
	sig := Sigmoid(*sw.SVal)
	grad = grad * 2 * sig * (1 - sig)
	// grad = grad * sw.Z * (1 - sw.Z/float64(n))
	*sw.SGrad += grad

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.WG.Top); i++ {
		j := (i - int(sw.Z) + n) % n
		sw.WG.Top[i].Grad += sw.Top[j].Grad*simj + sw.Top[(j-1+n)%n].Grad*(1-simj)
	}
}

type refocus struct {
	GammaVal  *float64
	GammaGrad *float64

	SW *shiftedWeighting

	TopVal  []float64
	TopGrad []float64

	g float64
}

func newRefocus(gammaVal *float64, gammaGrad *float64, sw *shiftedWeighting) *refocus {
	rf := refocus{
		GammaVal:  gammaVal,
		GammaGrad: gammaGrad,
		SW:        sw,
		TopVal:    make([]float64, len(sw.Top)),
		TopGrad:   make([]float64, len(sw.Top)),
		g:         math.Log(math.Exp(*gammaVal)+1) + 1,
	}
	var sum float64 = 0
	for i := 0; i < len(rf.TopVal); i++ {
		rf.TopVal[i] = math.Pow(sw.Top[i].Val, rf.g)
		sum += rf.TopVal[i]
	}
	for i := 0; i < len(rf.TopVal); i++ {
		rf.TopVal[i] = rf.TopVal[i] / sum
	}
	return &rf
}

func (rf *refocus) backwardSW() {
	var topGV float64 = 0
	for i, topV := range rf.TopVal {
		topGV += rf.TopGrad[i] * topV
	}
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		rf.SW.Top[i].Grad += (rf.TopGrad[i] - topGV) * rf.g / sw.Val * rf.TopVal[i]
	}
}

func (rf *refocus) backwardGamma() {
	lns := make([]float64, len(rf.SW.Top))
	var lnexp float64 = 0
	var s float64 = 0
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		lns[i] = math.Log(sw.Val)
		pow := math.Pow(sw.Val, rf.g)
		lnexp += lns[i] * pow
		s += pow
	}
	lnexps := lnexp / s
	var grad float64 = 0
	for i, topV := range rf.TopVal {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		grad += rf.TopGrad[i] * (topV * (lns[i] - lnexps))
	}
	grad = grad / (1 + math.Exp(-(*rf.GammaVal)))
	*rf.GammaGrad += grad
}

func (rf *refocus) Backward() {
	rf.backwardSW()
	rf.backwardGamma()
}

type memRead struct {
	W      *refocus
	Memory *writtenMemory

	TopVal  []float64
	TopGrad []float64
}

func newMemRead(w *refocus, memory *writtenMemory) *memRead {
	m := len(memory.TopVal) / memory.N
	r := memRead{
		W:       w,
		Memory:  memory,
		TopVal:  make([]float64, m),
		TopGrad: make([]float64, m),
	}

	weights := blas64.Vector{Inc: 1, Data: w.TopVal}
	mem := blas64.General{Rows: memory.N, Cols: m, Stride: m, Data: memory.TopVal}
	top := blas64.Vector{Inc: 1, Data: r.TopVal}
	blas64.Gemv(blas.Trans, 1, mem, weights, 1, top)

	return &r
}

func (r *memRead) Backward() {
	n := r.Memory.N
	m := len(r.Memory.TopVal) / n

	grad := blas64.Vector{Inc: 1, Data: r.TopGrad}
	memVal := blas64.General{Rows: n, Cols: m, Stride: m, Data: r.Memory.TopVal}
	weightsGrad := blas64.Vector{Inc: 1, Data: r.W.TopGrad}
	blas64.Gemv(blas.NoTrans, 1, memVal, grad, 1, weightsGrad)

	memGrad := blas64.General{Rows: n, Cols: m, Stride: m, Data: r.Memory.TopGrad}
	weights := blas64.Vector{Inc: 1, Data: r.W.TopVal}
	blas64.Ger(1, weights, grad, memGrad)
}

type writtenMemory struct {
	Ws    []*refocus
	Heads []*Head        // We actually need only the erase and add vectors.
	Mtm1  *writtenMemory // memory at time t-1

	N       int // memoryN
	TopVal  []float64
	TopGrad []float64

	erase    [][]float64
	add      [][]float64
	erasures []float64
}

func newWrittenMemory(ws []*refocus, heads []*Head, mtm1 *writtenMemory) *writtenMemory {
	n := mtm1.N
	m := len(mtm1.TopVal) / n
	wm := writtenMemory{
		Ws:    ws,
		Heads: heads,
		Mtm1:  mtm1,

		N:       mtm1.N,
		TopVal:  make([]float64, len(mtm1.TopVal)),
		TopGrad: make([]float64, len(mtm1.TopVal)),

		erase:    makeTensor2(len(heads), m),
		add:      makeTensor2(len(heads), m),
		erasures: make([]float64, len(mtm1.TopVal)),
	}
	for i, h := range wm.Heads {
		erase := wm.erase[i]
		add := wm.add[i]
		addVec := h.AddVal()
		for j, e := range h.EraseVal() {
			erase[j] = Sigmoid(e)
			add[j] = Sigmoid(addVec[j])
		}
	}

	copy(wm.erasures, mtm1.TopVal)
	we := make([]float64, n*m)
	weG := blas64.General{Rows: n, Cols: m, Stride: m, Data: we}
	for k, ws := range wm.Ws {
		weights := blas64.Vector{Inc: 1, Data: ws.TopVal}
		erase := blas64.Vector{Inc: 1, Data: wm.erase[k]}
		for i := range we {
			we[i] = 1
		}
		blas64.Ger(-1, weights, erase, weG)
		floats.Mul(wm.erasures, we)
	}

	copy(wm.TopVal, wm.erasures)
	topG := blas64.General{Rows: n, Cols: m, Stride: m, Data: wm.TopVal}
	for k, ws := range wm.Ws {
		weights := blas64.Vector{Inc: 1, Data: ws.TopVal}
		add := blas64.Vector{Inc: 1, Data: wm.add[k]}
		blas64.Ger(1, weights, add, topG)
	}

	return &wm
}

func (wm *writtenMemory) div1MWE(out []float64) {
	m := len(wm.TopVal) / wm.N
	for i, e := range wm.erasures {
		mwe := 1 - out[i]
		if math.Abs(mwe) > 1e-6 {
			out[i] = e / mwe
		} else {
			j := i / m
			k := i % m
			mtilt := wm.Mtm1.TopVal[j*m+k]
			for q, ws := range wm.Ws {
				if q == i {
					continue
				}
				mtilt = mtilt * (1 - ws.TopVal[j]*wm.erase[q][k])
			}
			out[i] = mtilt
		}
	}
}

func (wm *writtenMemory) backwardWErase() {
	n := wm.N
	m := len(wm.TopVal) / n

	mgrad := make([]float64, n*m)
	mGradG := blas64.General{Rows: n, Cols: m, Stride: m, Data: mgrad}
	hEraseGrad := blas64.Vector{Inc: 1, Data: make([]float64, m)}
	for i, weights := range wm.Ws {
		erase := wm.erase[i]
		add := wm.add[i]
		eraseV := blas64.Vector{Inc: 1, Data: erase}
		addV := blas64.Vector{Inc: 1, Data: add}
		weightsVal := blas64.Vector{Inc: 1, Data: weights.TopVal}

		for j := range mgrad {
			mgrad[j] = 0
		}
		blas64.Ger(1, weightsVal, eraseV, mGradG)
		wm.div1MWE(mgrad)
		floats.Mul(mgrad, wm.TopGrad)

		weightsV := blas64.Vector{Inc: 1, Data: weights.TopGrad}
		blas64.Gemv(blas.NoTrans, -1, mGradG, eraseV, 1, weightsV)
		blas64.Gemv(blas.NoTrans, 1, blas64.General{Rows: n, Cols: m, Stride: m, Data: wm.TopGrad}, addV, 1, weightsV)

		hErase := wm.Heads[i].EraseGrad()
		for j := range hEraseGrad.Data {
			hEraseGrad.Data[j] = 0
		}
		blas64.Gemv(blas.Trans, -1, mGradG, weightsVal, 1, hEraseGrad)
		for j, e := range erase {
			hErase[j] += hEraseGrad.Data[j] * e * (1 - e)
		}
	}
}

func (wm *writtenMemory) backwardAdd() {
	n := wm.N
	m := len(wm.TopVal) / n

	var grad float64
	for k, h := range wm.Heads {
		add := wm.add[k]
		ws := wm.Ws[k]
		hAdd := h.AddGrad()
		for i := range hAdd {
			grad = 0
			for j := 0; j < n; j++ {
				grad += wm.TopGrad[j*m+i] * ws.TopVal[j]
			}
			a := add[i]
			hAdd[i] += grad * a * (1 - a)
		}
	}
}

func (wm *writtenMemory) backwardMtm1() {
	n := wm.N
	m := len(wm.TopVal) / n

	var grad float64
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			grad = 1
			for q, ws := range wm.Ws {
				grad = grad * (1 - ws.TopVal[i]*wm.erase[q][j])
			}
			wm.Mtm1.TopGrad[i*m+j] += grad * wm.TopGrad[i*m+j]
		}
	}
}

func (wm *writtenMemory) Backward() {
	wm.backwardWErase()
	wm.backwardAdd()
	wm.backwardMtm1()
}

type memOp struct {
	W  []*refocus
	R  []*memRead
	WM *writtenMemory
}

func newMemOp(heads []*Head, mtm1 *writtenMemory) *memOp {
	circuit := memOp{
		R: make([]*memRead, len(heads)),
	}
	circuit.W = make([]*refocus, len(heads))
	for wi, h := range heads {
		ss := make([]*betaSimilarity, mtm1.N)
		for i := 0; i < mtm1.N; i++ {
			m := len(mtm1.TopVal) / mtm1.N
			s := newSimilarityCircuit(h.KVal(), h.KGrad(), mtm1.TopVal[i*m:(i+1)*m], mtm1.TopGrad[i*m:(i+1)*m])
			ss[i] = newBetaSimilarity(h.BetaVal(), h.BetaGrad(), s)
		}
		wc := newContentAddressing(ss)
		wg := newGatedWeighting(h.GVal(), h.GGrad(), wc, h.Wtm1)
		ws := newShiftedWeighting(h.SVal(), h.SGrad(), wg)
		circuit.W[wi] = newRefocus(h.GammaVal(), h.GammaGrad(), ws)
		circuit.R[wi] = newMemRead(circuit.W[wi], mtm1)
	}

	circuit.WM = newWrittenMemory(circuit.W, heads, mtm1)
	return &circuit
}

func (c *memOp) Backward() {
	for _, r := range c.R {
		r.Backward()
	}
	c.WM.Backward()

	for _, rf := range c.WM.Ws {
		rf.Backward()
		rf.SW.Backward()
		rf.SW.WG.Backward()
		rf.SW.WG.WC.Backward()
		for _, bs := range rf.SW.WG.WC.Units {
			bs.Backward()
			bs.S.Backward()
		}
	}
}
