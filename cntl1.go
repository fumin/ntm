package ntm

import (
	"fmt"
)

type Controller1 struct {
	wtm1s      [][]*BetaSimilarity
	mtm1       *WrittenMemory
	Wh1r       [][][]Unit
	Wh1x       [][]Unit
	Wh1b       []Unit
	Wyh1       [][]Unit
	Wuh1       [][][]Unit
	numWeights int

	Reads []*Read
	X     []float64

	H1 []Unit

	y     []Unit
	heads []*Head
}

func NewEmptyController1(xSize, ySize, h1Size, numHeads, n, m int) *Controller1 {
	h := NewHead(m)
	headUnitsSize := len(h.units)
	c := Controller1{
		wtm1s: make([][]*BetaSimilarity, numHeads),
		mtm1:  &WrittenMemory{Top: makeTensorUnit2(n, m)},
		Wh1r:  makeTensorUnit3(h1Size, numHeads, m),
		Wh1x:  makeTensorUnit2(h1Size, xSize),
		Wh1b:  make([]Unit, h1Size),
		Wyh1:  makeTensorUnit2(ySize, h1Size+1),
		Wuh1:  makeTensorUnit3(numHeads, headUnitsSize, h1Size+1),
	}
	for i := range c.wtm1s {
		c.wtm1s[i] = make([]*BetaSimilarity, n)
		for j := range c.wtm1s[i] {
			c.wtm1s[i][j] = &BetaSimilarity{}
		}
	}
	c.numWeights = numHeads*n + n*m + h1Size*numHeads*m + h1Size*xSize + h1Size + ySize*(h1Size+1) + numHeads*headUnitsSize*(h1Size+1)
	return &c
}

func (c *Controller1) Heads() []*Head {
	return c.heads
}

func (c *Controller1) Y() []Unit {
	return c.y
}

func (old *Controller1) Forward(reads []*Read, x []float64) Controller {
	c := Controller1{
		Wh1r:       old.Wh1r,
		Wh1x:       old.Wh1x,
		Wh1b:       old.Wh1b,
		Wyh1:       old.Wyh1,
		Wuh1:       old.Wuh1,
		numWeights: old.numWeights,
		Reads:      reads,
		X:          x,
		H1:         make([]Unit, len(old.Wh1r)),
		y:          make([]Unit, len(old.Wyh1)),
		heads:      make([]*Head, len(reads)),
	}

	var v float64
	for i, wh1ri := range c.Wh1r {
		wh1xi := c.Wh1x[i]
		v = 0
		for j, wh1rij := range wh1ri {
			read := reads[j]
			for k, wh1rijk := range wh1rij {
				v += wh1rijk.Val * read.Top[k].Val
			}
		}
		for j, wh1xij := range wh1xi {
			v += wh1xij.Val * x[j]
		}
		v += c.Wh1b[i].Val
		c.H1[i].Val = Sigmoid(v)
	}

	for i, wyh1i := range c.Wyh1 {
		v = 0
		for j, wyh1ij := range wyh1i[0:len(c.H1)] {
			v += wyh1ij.Val * c.H1[j].Val
		}
		v += c.Wyh1[i][len(c.H1)].Val
		c.y[i].Val = Sigmoid(v)
	}
	memoryM := len(reads[0].Top)
	for i, wuh1i := range c.Wuh1 {
		c.heads[i] = NewHead(memoryM)
		head := c.heads[i]
		for j, wuh1ij := range wuh1i {
			v = 0
			for k, wuh1ijk := range wuh1ij[0:len(c.H1)] {
				v += wuh1ijk.Val * c.H1[k].Val
			}
			v += wuh1ij[len(c.H1)].Val
			head.units[j].Val += v
		}
	}

	return &c
}

func (c *Controller1) Backward() {
	for j, y := range c.y {
		for i, wyh1 := range c.Wyh1[j][0:len(c.H1)] {
			c.H1[i].Grad += wyh1.Val * y.Grad
		}
	}
	for j, head := range c.heads {
		wuh1j := c.Wuh1[j]
		for k, h := range head.units {
			for i, wuh1jki := range wuh1j[k][0:len(c.H1)] {
				c.H1[i].Grad += h.Grad * wuh1jki.Val
			}
		}
	}
	for i, wyh1i := range c.Wyh1 {
		yGrad := c.y[i].Grad
		for j, h1 := range c.H1 {
			wyh1i[j].Grad += yGrad * h1.Val
		}
		wyh1i[len(wyh1i)-1].Grad += yGrad
	}
	for i, wuh1i := range c.Wuh1 {
		for j, head := range c.heads[i].units {
			wuh1ij := wuh1i[j]
			for k, h1 := range c.H1 {
				wuh1ij[k].Grad += head.Grad * h1.Val
			}
			wuh1ij[len(wuh1ij)-1].Grad += head.Grad
		}
	}

	h1Grads := make([]float64, len(c.H1))
	for i, h1 := range c.H1 {
		h1Grads[i] = h1.Grad * h1.Val * (1 - h1.Val)
	}

	for k, h1g := range h1Grads {
		wh1rk := c.Wh1r[k]
		for i, read := range c.Reads {
			wh1rki := wh1rk[i]
			for j, wh1rkij := range wh1rki {
				read.Top[j].Grad += h1g * wh1rkij.Val
			}
		}
	}
	for i, wh1ri := range c.Wh1r {
		h1g := h1Grads[i]
		for j, wh1rij := range wh1ri {
			for k, read := range c.Reads[j].Top {
				wh1rij[k].Grad += h1g * read.Val
			}
		}
	}
	for i, wh1xi := range c.Wh1x {
		h1g := h1Grads[i]
		for j, x := range c.X {
			wh1xi[j].Grad += h1g * x
		}
	}
	for i, h1g := range h1Grads {
		c.Wh1b[i].Grad += h1g
	}
}

func (c *Controller1) Wtm1BiasV() [][]*BetaSimilarity {
	return c.wtm1s
}

func (c *Controller1) Mtm1BiasV() *WrittenMemory {
	return c.mtm1
}

func (c *Controller1) Weights(f func(*Unit)) {
	for _, wtm1 := range c.wtm1s {
		for _, w := range wtm1 {
			f(&w.Top)
		}
	}
	for _, row := range c.mtm1.Top {
		for i := range row {
			f(&row[i])
		}
	}
	doUnit2(c.Wyh1, func(ids []int, u *Unit) { f(u) })
	doUnit3(c.Wuh1, func(ids []int, u *Unit) { f(u) })
	doUnit3(c.Wh1r, func(ids []int, u *Unit) { f(u) })
	doUnit2(c.Wh1x, func(ids []int, u *Unit) { f(u) })
	doUnit1(c.Wh1b, func(ids []int, u *Unit) { f(u) })
}

// WeightsVerbose is similar to Weights, but with additional information passed in.
// Avoid using this function except for debugging, as it calls fmt.Sprintf many times which is a performance hog.
func (c *Controller1) WeightsVerbose(f func(string, *Unit)) {
	for i, wtm1 := range c.wtm1s {
		for j, w := range wtm1 {
			f(fmt.Sprintf("wtm1[%d][%d]", i, j), &w.Top)
		}
	}
	for i, row := range c.mtm1.Top {
		for j := range row {
			f(fmt.Sprintf("mtm1[%d][%d]", i, j), &row[j])
		}
	}
	tagify := func(tag string, ids []int) string {
		s := tag
		for i := len(ids) - 1; i >= 0; i-- {
			s = fmt.Sprintf("%s[%d]", s, ids[i])
		}
		return s
	}
	doUnit2(c.Wyh1, func(ids []int, u *Unit) { f(tagify("Wyh1", ids), u) })
	doUnit3(c.Wuh1, func(ids []int, u *Unit) { f(tagify("Wuh1", ids), u) })
	doUnit3(c.Wh1r, func(ids []int, u *Unit) { f(tagify("Wh1r", ids), u) })
	doUnit2(c.Wh1x, func(ids []int, u *Unit) { f(tagify("Wh1x", ids), u) })
	doUnit1(c.Wh1b, func(ids []int, u *Unit) { f(tagify("Wh1b", ids), u) })
}

func (c *Controller1) NumWeights() int {
	return c.numWeights
}

func (c *Controller1) NumHeads() int {
	return len(c.Wuh1)
}

func (c *Controller1) MemoryN() int {
	return len(c.mtm1.Top)
}

func (c *Controller1) MemoryM() int {
	return len(c.Wh1r[0][0])
}
