package ntm

type Controller1 struct {
	Wh1r       [][][]Unit
	Wh1x       [][]Unit
	Wyh1       [][]Unit
	Wuh1       [][][]Unit
	numWeights int

	Reads []*Read
	X     []float64

	H1 []Unit

	y     []Unit
	heads []*Head
}

func NewEmptyController1(xSize, ySize, h1Size, numHeads, m int) *Controller1 {
	h := NewHead(m)
	headUnitsSize := len(h.units)
	c := Controller1{
		Wh1r:       makeTensorUnit3(h1Size, numHeads, m),
		Wh1x:       makeTensorUnit2(h1Size, xSize),
		Wyh1:       makeTensorUnit2(ySize, h1Size),
		Wuh1:       makeTensorUnit3(numHeads, headUnitsSize, h1Size),
		numWeights: h1Size*numHeads*m + h1Size*xSize + ySize*h1Size + numHeads*headUnitsSize*h1Size,
	}
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
		c.H1[i].Val = sigmoid(v)
	}

	for i := 0; i < len(c.y); i++ {
		var v float64 = 0
		for j := 0; j < len(c.H1); j++ {
			v += c.Wyh1[i][j].Val * c.H1[j].Val
		}
		c.y[i].Val = sigmoid(v)
	}
	memoryM := len(reads[0].Top)
	for i := 0; i < len(c.heads); i++ {
		c.heads[i] = NewHead(memoryM)
		for j := 0; j < len(c.heads[i].units); j++ {
			for k := 0; k < len(c.Wuh1[i][j]); k++ {
				c.heads[i].units[j].Val += c.Wuh1[i][j][k].Val * c.H1[k].Val
			}
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
		for j := 0; j < len(c.Wyh1[i]); j++ {
			c.Wyh1[i][j].Grad += c.y[i].Grad * c.H1[j].Val
		}
	}
	for i := 0; i < len(c.Wuh1); i++ {
		for j := 0; j < len(c.Wuh1[i]); j++ {
			for k := 0; k < len(c.Wuh1[i][j]); k++ {
				c.Wuh1[i][j][k].Grad += c.heads[i].units[j].Grad * c.H1[k].Val
			}
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
}

func (c *Controller1) Weights() chan *Unit {
	cu := make(chan *Unit)
	go func() {
		defer close(cu)
		for i := 0; i < len(c.Wh1r); i++ {
			for j := 0; j < len(c.Wh1r[i]); j++ {
				for k := 0; k < len(c.Wh1r[i][j]); k++ {
					cu <- &c.Wh1r[i][j][k]
				}
			}
		}
		for i := 0; i < len(c.Wh1x); i++ {
			for j := 0; j < len(c.Wh1x[i]); j++ {
				cu <- &c.Wh1x[i][j]
			}
		}
		for i := 0; i < len(c.Wyh1); i++ {
			for j := 0; j < len(c.Wyh1[i]); j++ {
				cu <- &c.Wyh1[i][j]
			}
		}
		for i := 0; i < len(c.Wuh1); i++ {
			for j := 0; j < len(c.Wuh1[i]); j++ {
				for k := 0; k < len(c.Wuh1[i][j]); k++ {
					cu <- &c.Wuh1[i][j][k]
				}
			}
		}
	}()
	return cu
}

func (c *Controller1) ClearGradients() {
	clearGrad3(c.Wh1r)
	clearGrad2(c.Wh1x)
	clearGrad2(c.Wyh1)
	clearGrad3(c.Wuh1)
}

func (c *Controller1) NumWeights() int {
	return c.numWeights
}

func (c *Controller1) NumHeads() int {
	return len(c.Wuh1)
}

func (c *Controller1) MemoryM() int {
	return len(c.Wh1r[0][0])
}
