package ntm

import (
	"fmt"
	"log"
	"math"
)

type Similarity struct {
	U   []Unit
	V   []Unit
	Top Unit

	UV    float64
	Unorm float64
	Vnorm float64
}

func NewSimilarity(u, v []Unit) *Similarity {
	s := Similarity{
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

func (s *Similarity) Backward() {
	uvuu := s.UV / (s.Unorm * s.Unorm)
	uvvv := s.UV / (s.Vnorm * s.Vnorm)
	uvg := s.Top.Grad / (s.Unorm * s.Vnorm)
	for i, u := range s.U {
		v := s.V[i].Val
		s.U[i].Grad += (v - u.Val*uvuu) * uvg
		s.V[i].Grad += (u.Val - v*uvvv) * uvg
	}
}

type BetaSimilarity struct {
	Beta *Unit // Beta is assumed to be in the range (-Inf, Inf)
	S    *Similarity
	Top  Unit

	b float64
}

func NewBetaSimilarity(beta *Unit, s *Similarity) *BetaSimilarity {
	bs := BetaSimilarity{
		Beta: beta,
		S:    s,
		b:    math.Exp(beta.Val),
	}
	bs.Top.Val = bs.b * s.Top.Val
	return &bs
}

func (bs *BetaSimilarity) Backward() {
	bs.Beta.Grad += bs.S.Top.Val * bs.b * bs.Top.Grad
	bs.S.Top.Grad += bs.b * bs.Top.Grad
}

type ContentAddressing struct {
	Units []*BetaSimilarity
	Top   []Unit
}

func NewContentAddressing(units []*BetaSimilarity) *ContentAddressing {
	s := ContentAddressing{
		Units: units,
		Top:   make([]Unit, len(units)),
	}
	// Increase numerical stability by subtracting all weights by their max,
	// before computing math.Exp().
	var max float64 = -1
	for _, u := range s.Units {
		max = math.Max(max, u.Top.Val)
	}
	var sum float64 = 0
	for i := 0; i < len(s.Top); i++ {
		s.Top[i].Val = math.Exp(s.Units[i].Top.Val - max)
		sum += s.Top[i].Val
	}
	for i := 0; i < len(s.Top); i++ {
		s.Top[i].Val = s.Top[i].Val / sum
	}
	return &s
}

func (s *ContentAddressing) Backward() {
	var gv float64 = 0
	for _, top := range s.Top {
		gv += top.Grad * top.Val
	}
	for i, top := range s.Top {
		s.Units[i].Top.Grad += (top.Grad - gv) * top.Val
	}
}

type GatedWeighting struct {
	G    *Unit
	WC   *ContentAddressing
	Wtm1 *Refocus // the weights at time t-1
	Top  []Unit
}

func NewGatedWeighting(g *Unit, wc *ContentAddressing, wtm1 *Refocus) *GatedWeighting {
	wg := GatedWeighting{
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

func (wg *GatedWeighting) Backward() {
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

type ShiftedWeighting struct {
	S   *Unit
	Z   float64
	WG  *GatedWeighting
	Top []Unit
}

func NewShiftedWeighting(s *Unit, wg *GatedWeighting) *ShiftedWeighting {
	sw := ShiftedWeighting{
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
		if sw.Top[i].Val < 0 {
			log.Printf("imj: %d, wg: %f, simj: %f, wg+1: %f", imj, sw.WG.Top[imj].Val, simj, sw.WG.Top[(imj+1)%n].Val)
			panic("")
		}
	}
	return &sw
}

func (sw *ShiftedWeighting) Backward() {
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

type Refocus struct {
	Gamma *Unit
	SW    *ShiftedWeighting
	Top   []Unit

	g float64
}

func NewRefocus(gamma *Unit, sw *ShiftedWeighting) *Refocus {
	rf := Refocus{
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

func (rf *Refocus) Backward() {
	for i := 0; i < len(rf.SW.Top); i++ {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		var grad float64 = 0
		for j := 0; j < len(rf.Top); j++ {
			if j == i {
				grad += rf.Top[j].Grad * (1 - rf.Top[j].Val)
			} else {
				grad -= rf.Top[j].Grad * rf.Top[j].Val
			}
		}
		grad = grad * rf.g / rf.SW.Top[i].Val * rf.Top[i].Val
		rf.SW.Top[i].Grad += grad
	}

	lns := make([]float64, len(rf.SW.Top))
	var lnexp float64 = 0
	var s float64 = 0
	for i := 0; i < len(lns); i++ {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		lns[i] = math.Log(rf.SW.Top[i].Val)
		pow := math.Pow(rf.SW.Top[i].Val, rf.g)
		lnexp += lns[i] * pow
		s += pow
	}
	lnexps := lnexp / s
	var grad float64 = 0
	for i := 0; i < len(rf.Top); i++ {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		grad += rf.Top[i].Grad * (rf.Top[i].Val * (lns[i] - lnexps))
	}
	grad = grad / (1 + math.Exp(-rf.Gamma.Val))
	rf.Gamma.Grad += grad
}

type Read struct {
	W      *Refocus
	Memory *WrittenMemory
	Top    []Unit
}

func NewRead(w *Refocus, memory *WrittenMemory) *Read {
	r := Read{
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

func (r *Read) Backward() {
	for i := 0; i < len(r.W.Top); i++ {
		var grad float64 = 0
		for j := 0; j < len(r.Top); j++ {
			grad += r.Top[j].Grad * r.Memory.Top[i][j].Val
		}
		r.W.Top[i].Grad += grad
	}

	for i := 0; i < len(r.Memory.Top); i++ {
		for j := 0; j < len(r.Memory.Top[i]); j++ {
			r.Memory.Top[i][j].Grad += r.Top[j].Grad * r.W.Top[i].Val
		}
	}
}

type WrittenMemory struct {
	Ws    []*Refocus
	Heads []*Head        // We actually need only the erase and add vectors.
	Mtm1  *WrittenMemory // memory at time t-1
	Top   [][]Unit

	erase [][]float64
	add   [][]float64
	mTilt [][]float64
}

func NewWrittenMemory(ws []*Refocus, heads []*Head, mtm1 *WrittenMemory) *WrittenMemory {
	wm := WrittenMemory{
		Ws:    ws,
		Heads: heads,
		Mtm1:  mtm1,
		Top:   makeTensorUnit2(len(mtm1.Top), len(mtm1.Top[0])),

		erase: MakeTensor2(len(heads), len(mtm1.Top[0])),
		add:   MakeTensor2(len(heads), len(mtm1.Top[0])),
		mTilt: MakeTensor2(len(mtm1.Top), len(mtm1.Top[0])),
	}
	for i := 0; i < len(wm.Heads); i++ {
		eraseVec := wm.Heads[i].EraseVector()
		addVec := wm.Heads[i].AddVector()
		for j := 0; j < len(wm.erase[i]); j++ {
			wm.erase[i][j] = Sigmoid(eraseVec[j].Val)
			wm.add[i][j] = Sigmoid(addVec[j].Val)
		}
	}

	for i := 0; i < len(wm.Top); i++ {
		for j := 0; j < len(wm.Top[i]); j++ {
			var mtilt float64 = 1
			var adds float64 = 0
			for k := 0; k < len(wm.Heads); k++ {
				mtilt = mtilt * (1 - wm.Ws[k].Top[i].Val*wm.erase[k][j])
				adds = adds + wm.Ws[k].Top[i].Val*wm.add[k][j]
			}
			wm.mTilt[i][j] = wm.Mtm1.Top[i][j].Val * mtilt
			wm.Top[i][j].Val = wm.mTilt[i][j] + adds
		}
	}
	return &wm
}

func (wm *WrittenMemory) Backward() {
	// Gradient of W
	for i := 0; i < len(wm.Ws); i++ {
		for j := 0; j < len(wm.Ws[i].Top); j++ {
			var grad float64 = 0
			for k := 0; k < len(wm.Top[j]); k++ {
				e := wm.erase[i][k]
				gErase := wm.Mtm1.Top[j][k].Val * (-e)
				for q := 0; q < len(wm.Ws); q++ {
					if q == i {
						continue
					}
					gErase = gErase * (1 - wm.Ws[q].Top[j].Val*wm.erase[q][k])
				}
				gAdd := wm.add[i][k]
				grad += (gErase + gAdd) * wm.Top[j][k].Grad
			}
			wm.Ws[i].Top[j].Grad += grad
		}
	}

	// Gradient of Erase vector
	for k, h := range wm.Heads {
		for i := 0; i < len(h.EraseVector()); i++ {
			var grad float64 = 0
			for j := 0; j < len(wm.Top); j++ {
				gErase := wm.Mtm1.Top[j][i].Val
				for q := 0; q < len(wm.Ws); q++ {
					if q == k {
						continue
					}
					gErase = gErase * (1 - wm.Ws[q].Top[j].Val*wm.erase[q][i])
				}
				grad += wm.Top[j][i].Grad * gErase * (-wm.Ws[k].Top[j].Val)
			}
			h.EraseVector()[i].Grad += grad * wm.erase[k][i] * (1 - wm.erase[k][i])
		}
	}

	// Gradient of Add vector
	for k, h := range wm.Heads {
		for i := 0; i < len(h.AddVector()); i++ {
			var grad float64 = 0
			for j := 0; j < len(wm.Top); j++ {
				grad += wm.Top[j][i].Grad * wm.Ws[k].Top[j].Val
			}
			h.AddVector()[i].Grad += grad * wm.add[k][i] * (1 - wm.add[k][i])
		}
	}

	// Gradient of wm.Mtm1
	for i := 0; i < len(wm.Mtm1.Top); i++ {
		for j := 0; j < len(wm.Mtm1.Top[i]); j++ {
			var grad float64 = 1
			for q := 0; q < len(wm.Ws); q++ {
				grad = grad * (1 - wm.Ws[q].Top[i].Val*wm.erase[q][j])
			}
			wm.Mtm1.Top[i][j].Grad += grad * wm.Top[i][j].Grad
		}
	}
}

type Circuit struct {
	W  []*Refocus
	R  []*Read
	WM *WrittenMemory
}

func NewCircuit(heads []*Head, mtm1 *WrittenMemory) *Circuit {
	circuit := Circuit{
		R: make([]*Read, len(heads)),
	}
	circuit.W = make([]*Refocus, len(heads))
	for wi, h := range heads {
		ss := make([]*BetaSimilarity, len(mtm1.Top))
		for i := 0; i < len(mtm1.Top); i++ {
			s := NewSimilarity(h.K(), mtm1.Top[i])
			ss[i] = NewBetaSimilarity(h.Beta(), s)
		}
		wc := NewContentAddressing(ss)
		wg := NewGatedWeighting(h.G(), wc, h.Wtm1)
		ws := NewShiftedWeighting(h.S(), wg)
		circuit.W[wi] = NewRefocus(h.Gamma(), ws)
		circuit.R[wi] = NewRead(circuit.W[wi], mtm1)
	}

	circuit.WM = NewWrittenMemory(circuit.W, heads, mtm1)
	return &circuit
}

func (c *Circuit) Backward() {
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

func (c *Circuit) ReadVals() [][]float64 {
	res := MakeTensor2(len(c.R), len(c.R[0].Top))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.R[i].Top[j].Val
		}
	}
	return res
}

func (c *Circuit) WrittenMemoryVals() [][]float64 {
	res := MakeTensor2(len(c.WM.Top), len(c.WM.Top[0]))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.WM.Top[i][j].Val
		}
	}
	return res
}
