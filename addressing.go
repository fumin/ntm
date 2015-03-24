package ntm

import (
	"fmt"
	"log"
	"math"
)

type similarityCircuit struct {
	U   []Unit
	V   []Unit
	Top Unit

	UV    float64
	Unorm float64
	Vnorm float64
}

func newSimilarityCircuit(u, v []Unit) *similarityCircuit {
	s := similarityCircuit{
		U: u,
		V: v,
	}
	for i := 0; i < len(u); i++ {
		s.UV += u[i].Val * v[i].Val
		s.Unorm += u[i].Val * u[i].Val
		s.Vnorm += v[i].Val * v[i].Val
	}
	s.Unorm = math.Sqrt(s.Unorm)
	s.Vnorm = math.Sqrt(s.Vnorm)
	s.Top.Val = s.UV / (s.Unorm * s.Vnorm)
	if math.IsNaN(s.Top.Val) {
		log.Printf("u: %+v, v: %+v", u, v)
		panic("")
	}
	return &s
}

func (s *similarityCircuit) Backward() {
	uvuu := s.UV / (s.Unorm * s.Unorm)
	uvvv := s.UV / (s.Vnorm * s.Vnorm)
	uvg := s.Top.Grad / (s.Unorm * s.Vnorm)
	for i, u := range s.U {
		v := s.V[i].Val
		s.U[i].Grad += (v - u.Val*uvuu) * uvg
		s.V[i].Grad += (u.Val - v*uvvv) * uvg
	}
}

type betaSimilarity struct {
	Beta *Unit // Beta is assumed to be in the range (-Inf, Inf)
	S    *similarityCircuit
	Top  Unit

	b float64
}

func newBetaSimilarity(beta *Unit, s *similarityCircuit) *betaSimilarity {
	bs := betaSimilarity{
		Beta: beta,
		S:    s,
		b:    math.Exp(beta.Val),
	}
	bs.Top.Val = bs.b * s.Top.Val
	return &bs
}

func (bs *betaSimilarity) Backward() {
	bs.Beta.Grad += bs.S.Top.Val * bs.b * bs.Top.Grad
	bs.S.Top.Grad += bs.b * bs.Top.Grad
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
	G    *Unit
	WC   *contentAddressing
	Wtm1 *refocus // the weights at time t-1
	Top  []Unit
}

func newGatedWeighting(g *Unit, wc *contentAddressing, wtm1 *refocus) *gatedWeighting {
	wg := gatedWeighting{
		G:    g,
		WC:   wc,
		Wtm1: wtm1,
		Top:  make([]Unit, len(wc.Top)),
	}
	gt := Sigmoid(g.Val)
	for i := 0; i < len(wg.Top); i++ {
		wg.Top[i].Val = gt*wc.Top[i].Val + (1-gt)*wtm1.Top[i].Val
	}
	return &wg
}

func (wg *gatedWeighting) Backward() {
	gt := Sigmoid(wg.G.Val)

	var grad float64 = 0
	for i := 0; i < len(wg.Top); i++ {
		grad += (wg.WC.Top[i].Val - wg.Wtm1.Top[i].Val) * wg.Top[i].Grad
	}
	wg.G.Grad += grad * gt * (1 - gt)

	for i := 0; i < len(wg.WC.Top); i++ {
		wg.WC.Top[i].Grad += gt * wg.Top[i].Grad
	}

	for i := 0; i < len(wg.Wtm1.Top); i++ {
		wg.Wtm1.Top[i].Grad += (1 - gt) * wg.Top[i].Grad
	}
}

type shiftedWeighting struct {
	S   *Unit
	Z   float64
	WG  *gatedWeighting
	Top []Unit
}

func newShiftedWeighting(s *Unit, wg *gatedWeighting) *shiftedWeighting {
	sw := shiftedWeighting{
		S:   s,
		WG:  wg,
		Top: make([]Unit, len(wg.Top)),
	}

	n := len(sw.WG.Top)
	//sw.Z = math.Mod(s.Val, float64(n))
	//if sw.Z < 0 {
	//	sw.Z += float64(n)
	//}

	//sw.Z = float64(n) * Sigmoid(s.Val)
	shift := (2*Sigmoid(s.Val) - 1) // * maxShift
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
	sig := Sigmoid(sw.S.Val)
	grad = grad * 2 * sig * (1 - sig)
	// grad = grad * sw.Z * (1 - sw.Z/float64(n))
	sw.S.Grad += grad

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.WG.Top); i++ {
		j := (i - int(sw.Z) + n) % n
		sw.WG.Top[i].Grad += sw.Top[j].Grad*simj + sw.Top[(j-1+n)%n].Grad*(1-simj)
	}
}

type refocus struct {
	Gamma *Unit
	SW    *shiftedWeighting
	Top   []Unit

	g float64
}

func newRefocus(gamma *Unit, sw *shiftedWeighting) *refocus {
	rf := refocus{
		Gamma: gamma,
		SW:    sw,
		Top:   make([]Unit, len(sw.Top)),
		g:     math.Log(math.Exp(gamma.Val)+1) + 1,
	}
	var sum float64 = 0
	for i := 0; i < len(rf.Top); i++ {
		rf.Top[i].Val = math.Pow(sw.Top[i].Val, rf.g)
		sum += rf.Top[i].Val
	}
	for i := 0; i < len(rf.Top); i++ {
		rf.Top[i].Val = rf.Top[i].Val / sum
		if math.IsNaN(rf.Top[i].Val) {
			log.Printf("g: %f, sw: %+v", rf.g, sw.Top)
			panic(fmt.Sprintf("rf: %f, sum: %f", rf.Top[i].Val, sum))
		}
	}
	return &rf
}

func (rf *refocus) backwardSW() {
	var topGV float64 = 0
	for _, top := range rf.Top {
		topGV += top.Grad * top.Val
	}
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		rf.SW.Top[i].Grad += (rf.Top[i].Grad - topGV) * rf.g / sw.Val * rf.Top[i].Val
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
	for i, top := range rf.Top {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		grad += top.Grad * (top.Val * (lns[i] - lnexps))
	}
	grad = grad / (1 + math.Exp(-rf.Gamma.Val))
	rf.Gamma.Grad += grad
}

func (rf *refocus) Backward() {
	rf.backwardSW()
	rf.backwardGamma()
}

type memRead struct {
	W      *refocus
	Memory *writtenMemory
	Top    []Unit
}

func newMemRead(w *refocus, memory *writtenMemory) *memRead {
	r := memRead{
		W:      w,
		Memory: memory,
		Top:    make([]Unit, len(memory.Top[0])),
	}
	for i := 0; i < len(r.Top); i++ {
		var v float64 = 0
		for j := 0; j < len(w.Top); j++ {
			v += w.Top[j].Val * memory.Top[j][i].Val
			if math.IsNaN(v) {
				panic(fmt.Sprintf("w: %f, mem: %f", w.Top[j].Val, memory.Top[j][i].Val))
			}
		}
		r.Top[i].Val = v
	}
	return &r
}

func (r *memRead) Backward() {
	for i, memI := range r.Memory.Top {
		var grad float64 = 0
		for j, mem := range memI {
			grad += r.Top[j].Grad * mem.Val
		}
		r.W.Top[i].Grad += grad
	}

	for i, memI := range r.Memory.Top {
		w := r.W.Top[i].Val
		for j, top := range r.Top {
			memI[j].Grad += top.Grad * w
		}
	}
}

type writtenMemory struct {
	Ws    []*refocus
	Heads []*Head        // We actually need only the erase and add vectors.
	Mtm1  *writtenMemory // memory at time t-1
	Top   [][]Unit

	erase    [][]float64
	add      [][]float64
	erasures [][]float64
}

func newWrittenMemory(ws []*refocus, heads []*Head, mtm1 *writtenMemory) *writtenMemory {
	wm := writtenMemory{
		Ws:    ws,
		Heads: heads,
		Mtm1:  mtm1,
		Top:   makeTensorUnit2(len(mtm1.Top), len(mtm1.Top[0])),

		erase:    MakeTensor2(len(heads), len(mtm1.Top[0])),
		add:      MakeTensor2(len(heads), len(mtm1.Top[0])),
		erasures: MakeTensor2(len(mtm1.Top), len(mtm1.Top[0])),
	}
	for i, h := range wm.Heads {
		erase := wm.erase[i]
		add := wm.add[i]
		eraseVec := h.EraseVector()
		addVec := h.AddVector()
		for j, e := range eraseVec {
			erase[j] = Sigmoid(e.Val)
			add[j] = Sigmoid(addVec[j].Val)
		}
	}

	for i, mtm1Row := range wm.Mtm1.Top {
		erasure := wm.erasures[i]
		topRow := wm.Top[i]
		for j, mtm1 := range mtm1Row {
			var e float64 = 1
			var adds float64 = 0
			for k, weights := range wm.Ws {
				e = e * (1 - weights.Top[i].Val*wm.erase[k][j])
				adds += weights.Top[i].Val * wm.add[k][j]
			}
			erasure[j] = e
			topRow[j].Val += e*mtm1.Val + adds
		}
	}
	return &wm
}

func (wm *writtenMemory) backwardWErase() {
	var grad float64 = 0
	for i, weights := range wm.Ws {
		hErase := wm.Heads[i].EraseVector()
		erase := wm.erase[i]
		add := wm.add[i]
		ws := wm.Ws[i]
		for j, topRow := range wm.Top {
			mtm1Row := wm.Mtm1.Top[j]
			erasure := wm.erasures[j]
			wsj := ws.Top[j].Val
			grad = 0
			for k, mtm1 := range mtm1Row {
				mtilt := mtm1.Val
				e := erase[k]
				mwe := 1 - wsj*e
				if math.Abs(mwe) > 1e-6 {
					mtilt = mtilt * erasure[k] / mwe
				} else {
					for q, ws := range wm.Ws {
						if q == i {
							continue
						}
						mtilt = mtilt * (1 - ws.Top[j].Val*wm.erase[q][k])
					}
				}
				grad += (mtilt*(-e) + add[k]) * topRow[k].Grad
				hErase[k].Grad += topRow[k].Grad * mtilt * (-wsj)
			}
			weights.Top[j].Grad += grad
		}

		for j, e := range erase {
			hErase[j].Grad *= e * (1 - e)
		}
	}
}

func (wm *writtenMemory) backwardAdd() {
	var grad float64
	for k, h := range wm.Heads {
		add := wm.add[k]
		ws := wm.Ws[k]
		hAdd := h.AddVector()
		for i := range hAdd {
			grad = 0
			for j, toprow := range wm.Top {
				grad += toprow[i].Grad * ws.Top[j].Val
			}
			a := add[i]
			hAdd[i].Grad += grad * a * (1 - a)
		}
	}
}

func (wm *writtenMemory) backwardMtm1() {
	var grad float64
	for i, mtm1row := range wm.Mtm1.Top {
		toprow := wm.Top[i]
		for j, top := range toprow {
			grad = 1
			for q, ws := range wm.Ws {
				grad = grad * (1 - ws.Top[i].Val*wm.erase[q][j])
			}
			mtm1row[j].Grad += grad * top.Grad
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
		ss := make([]*betaSimilarity, len(mtm1.Top))
		for i := 0; i < len(mtm1.Top); i++ {
			s := newSimilarityCircuit(h.K(), mtm1.Top[i])
			ss[i] = newBetaSimilarity(h.Beta(), s)
		}
		wc := newContentAddressing(ss)
		wg := newGatedWeighting(h.G(), wc, h.Wtm1)
		ws := newShiftedWeighting(h.S(), wg)
		circuit.W[wi] = newRefocus(h.Gamma(), ws)
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

func (c *memOp) ReadVals() [][]float64 {
	res := MakeTensor2(len(c.R), len(c.R[0].Top))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.R[i].Top[j].Val
		}
	}
	return res
}

func (c *memOp) WrittenMemoryVals() [][]float64 {
	res := MakeTensor2(len(c.WM.Top), len(c.WM.Top[0]))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.WM.Top[i][j].Val
		}
	}
	return res
}
