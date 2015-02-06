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

	for i := 0; i < len(c.H1); i++ {
		var v float64 = 0
		for j := 0; j < len(reads); j++ {
			for k := 0; k < len(reads[j].Top); k++ {
				v += c.Wh1r[i][j][k].Val * reads[j].Top[k].Val
			}
		}
		for j := 0; j < len(x); j++ {
			v += c.Wh1x[i][j].Val * x[j]
		}
		v += c.Wh1b[i].Val
		c.H1[i].Val = sigmoid(v)
	}

	for i := 0; i < len(c.y); i++ {
		var v float64 = 0
		for j := 0; j < len(c.H1); j++ {
			v += c.Wyh1[i][j].Val * c.H1[j].Val
		}
		v += c.Wyh1[i][len(c.H1)].Val
		c.y[i].Val = sigmoid(v)
	}
	memoryM := len(reads[0].Top)
	for i := 0; i < len(c.heads); i++ {
		c.heads[i] = NewHead(memoryM)
		for j := 0; j < len(c.heads[i].units); j++ {
			maxK := len(c.Wuh1[i][j]) - 1
			for k := 0; k < maxK; k++ {
				c.heads[i].units[j].Val += c.Wuh1[i][j][k].Val * c.H1[k].Val
			}
			c.heads[i].units[j].Val += c.Wuh1[i][j][maxK].Val
		}
	}

	return &c
}

func (c *Controller1) Backward() {
	for i := 0; i < len(c.H1); i++ {
		var grad float64 = 0
		for j := 0; j < len(c.y); j++ {
			grad += c.y[j].Grad * c.Wyh1[j][i].Val
		}
		for j := 0; j < len(c.heads); j++ {
			for k := 0; k < len(c.heads[j].units); k++ {
				grad += c.heads[j].units[k].Grad * c.Wuh1[j][k][i].Val
			}
		}
		c.H1[i].Grad += grad
	}
	for i := 0; i < len(c.Wyh1); i++ {
		maxJ := len(c.Wyh1[i]) - 1
		for j := 0; j < maxJ; j++ {
			c.Wyh1[i][j].Grad += c.y[i].Grad * c.H1[j].Val
		}
		c.Wyh1[i][maxJ].Grad += c.y[i].Grad
	}
	for i := 0; i < len(c.Wuh1); i++ {
		for j := 0; j < len(c.Wuh1[i]); j++ {
			maxK := len(c.Wuh1[i][j]) - 1
			for k := 0; k < maxK; k++ {
				c.Wuh1[i][j][k].Grad += c.heads[i].units[j].Grad * c.H1[k].Val
			}
			c.Wuh1[i][j][maxK].Grad += c.heads[i].units[j].Grad
		}
	}

	for i := 0; i < len(c.Reads); i++ {
		for j := 0; j < len(c.Reads[i].Top); j++ {
			for k := 0; k < len(c.H1); k++ {
				c.Reads[i].Top[j].Grad += c.H1[k].Grad * c.H1[k].Val * (1 - c.H1[k].Val) * c.Wh1r[k][i][j].Val
			}
		}
	}
	for i := 0; i < len(c.Wh1r); i++ {
		for j := 0; j < len(c.Wh1r[i]); j++ {
			for k := 0; k < len(c.Wh1r[i][j]); k++ {
				c.Wh1r[i][j][k].Grad += c.H1[i].Grad * c.H1[i].Val * (1 - c.H1[i].Val) * c.Reads[j].Top[k].Val
			}
		}
	}
	for i := 0; i < len(c.Wh1x); i++ {
		for j := 0; j < len(c.Wh1x[i]); j++ {
			c.Wh1x[i][j].Grad += c.H1[i].Grad * c.H1[i].Val * (1 - c.H1[i].Val) * c.X[j]
		}
	}
	for i := 0; i < len(c.Wh1b); i++ {
		c.Wh1b[i].Grad += c.H1[i].Grad * c.H1[i].Val * (1 - c.H1[i].Val)
	}
}

func (c *Controller1) Wtm1BiasV() [][]*BetaSimilarity {
	return c.wtm1s
}

func (c *Controller1) Mtm1BiasV() *WrittenMemory {
	return c.mtm1
}

func (c *Controller1) Weights(f func(string, *Unit)) {
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
