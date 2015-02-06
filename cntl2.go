package ntm

//
//import (
//	"fmt"
//)
//
//type Controller2 struct {
//	WYh1r [][][]Unit
//	WYh1x [][]Unit
//	WYh1  [][]Unit
//
//	WUh1r      [][][]Unit
//	WUh1x      [][]Unit
//	WUh1       [][][]Unit
//	numWeights int
//
//	Reads []*Read
//	X     []float64
//	YH1   []Unit
//	UH1   []Unit
//	y     []Unit
//	heads []*Head
//}
//
//func NewEmptyController2(xSize, ySize, h1Size, numHeads, m int) *Controller2 {
//	h := NewHead(m)
//	headUnitsSize := len(h.units)
//	c := Controller2{
//		WYh1r:      makeTensorUnit3(h1Size, numHeads, m),
//		WYh1x:      makeTensorUnit2(h1Size, xSize),
//		WYh1:       makeTensorUnit2(ySize, h1Size),
//		WUh1r:      makeTensorUnit3(h1Size, numHeads, m),
//		WUh1x:      makeTensorUnit2(h1Size, xSize),
//		WUh1:       makeTensorUnit3(numHeads, headUnitsSize, h1Size),
//		numWeights: h1Size*numHeads*m + h1Size*xSize + ySize*h1Size + h1Size*numHeads*m + h1Size*xSize + numHeads*headUnitsSize*h1Size,
//	}
//	return &c
//}
//
//func (c *Controller2) Heads() []*Head {
//	return c.heads
//}
//
//func (c *Controller2) Y() []Unit {
//	return c.y
//}
//
//func (old *Controller2) Forward(reads []*Read, x []float64) Controller {
//	c := Controller2{
//		WYh1r:      old.WYh1r,
//		WYh1x:      old.WYh1x,
//		WYh1:       old.WYh1,
//		WUh1r:      old.WUh1r,
//		WUh1x:      old.WUh1x,
//		WUh1:       old.WUh1,
//		numWeights: old.numWeights,
//		Reads:      reads,
//		X:          x,
//		YH1:        make([]Unit, len(old.WYh1r)),
//		UH1:        make([]Unit, len(old.WUh1r)),
//		y:          make([]Unit, len(old.WYh1)),
//		heads:      make([]*Head, len(reads)),
//	}
//
//	for i := 0; i < len(c.YH1); i++ {
//		var v float64 = 0
//		for j := 0; j < len(reads); j++ {
//			for k := 0; k < len(reads[j].Top); k++ {
//				v += c.WYh1r[i][j][k].Val * reads[j].Top[k].Val
//			}
//		}
//		for j := 0; j < len(x); j++ {
//			v += c.WYh1x[i][j].Val * x[j]
//		}
//		c.YH1[i].Val = sigmoid(v)
//	}
//	for i := 0; i < len(c.y); i++ {
//		var v float64 = 0
//		for j := 0; j < len(c.YH1); j++ {
//			v += c.WYh1[i][j].Val * c.YH1[j].Val
//		}
//		c.y[i].Val = sigmoid(v)
//	}
//
//	for i := 0; i < len(c.UH1); i++ {
//		var v float64 = 0
//		for j := 0; j < len(reads); j++ {
//			for k := 0; k < len(reads[j].Top); k++ {
//				v += c.WUh1r[i][j][k].Val * reads[j].Top[k].Val
//			}
//		}
//		for j := 0; j < len(x); j++ {
//			v += c.WUh1x[i][j].Val * x[j]
//		}
//		c.UH1[i].Val = sigmoid(v)
//	}
//	memoryM := len(reads[0].Top)
//	for i := 0; i < len(c.heads); i++ {
//		c.heads[i] = NewHead(memoryM)
//		for j := 0; j < len(c.heads[i].units); j++ {
//			for k := 0; k < len(c.WUh1[i][j]); k++ {
//				c.heads[i].units[j].Val += c.WUh1[i][j][k].Val * c.UH1[k].Val
//			}
//		}
//	}
//
//	return &c
//}
//
//func (c *Controller2) Backward() {
//	for i := 0; i < len(c.YH1); i++ {
//		var grad float64 = 0
//		for j := 0; j < len(c.y); j++ {
//			grad += c.y[j].Grad * c.WYh1[j][i].Val
//		}
//		c.YH1[i].Grad += grad
//	}
//	for i := 0; i < len(c.WYh1); i++ {
//		for j := 0; j < len(c.WYh1[i]); j++ {
//			c.WYh1[i][j].Grad += c.y[i].Grad * c.YH1[j].Val
//		}
//	}
//	for i := 0; i < len(c.Reads); i++ {
//		for j := 0; j < len(c.Reads[i].Top); j++ {
//			for k := 0; k < len(c.YH1); k++ {
//				c.Reads[i].Top[j].Grad += c.YH1[k].Grad * c.YH1[k].Val * (1 - c.YH1[k].Val) * c.WYh1r[k][i][j].Val
//			}
//		}
//	}
//	for i := 0; i < len(c.WYh1r); i++ {
//		for j := 0; j < len(c.WYh1r[i]); j++ {
//			for k := 0; k < len(c.WYh1r[i][j]); k++ {
//				c.WYh1r[i][j][k].Grad += c.YH1[i].Grad * c.YH1[i].Val * (1 - c.YH1[i].Val) * c.Reads[j].Top[k].Val
//			}
//		}
//	}
//	for i := 0; i < len(c.WYh1x); i++ {
//		for j := 0; j < len(c.WYh1x[i]); j++ {
//			c.WYh1x[i][j].Grad += c.YH1[i].Grad * c.YH1[i].Val * (1 - c.YH1[i].Val) * c.X[j]
//		}
//	}
//
//	for i := 0; i < len(c.UH1); i++ {
//		var grad float64 = 0
//		for j := 0; j < len(c.heads); j++ {
//			for k := 0; k < len(c.heads[j].units); k++ {
//				grad += c.heads[j].units[k].Grad * c.WUh1[j][k][i].Val
//			}
//		}
//		c.UH1[i].Grad += grad
//	}
//	for i := 0; i < len(c.WUh1); i++ {
//		for j := 0; j < len(c.WUh1[i]); j++ {
//			for k := 0; k < len(c.WUh1[i][j]); k++ {
//				c.WUh1[i][j][k].Grad += c.heads[i].units[j].Grad * c.UH1[k].Val
//			}
//		}
//	}
//	for i := 0; i < len(c.Reads); i++ {
//		for j := 0; j < len(c.Reads[i].Top); j++ {
//			for k := 0; k < len(c.UH1); k++ {
//				c.Reads[i].Top[j].Grad += c.UH1[k].Grad * c.UH1[k].Val * (1 - c.UH1[k].Val) * c.WUh1r[k][i][j].Val
//			}
//		}
//	}
//	for i := 0; i < len(c.WUh1r); i++ {
//		for j := 0; j < len(c.WUh1r[i]); j++ {
//			for k := 0; k < len(c.WUh1r[i][j]); k++ {
//				c.WUh1r[i][j][k].Grad += c.UH1[i].Grad * c.UH1[i].Val * (1 - c.UH1[i].Val) * c.Reads[j].Top[k].Val
//			}
//		}
//	}
//	for i := 0; i < len(c.WUh1x); i++ {
//		for j := 0; j < len(c.WUh1x[i]); j++ {
//			c.WUh1x[i][j].Grad += c.UH1[i].Grad * c.UH1[i].Val * (1 - c.UH1[i].Val) * c.X[j]
//		}
//	}
//}
//
//func (c *Controller2) Weights(f func(string, *Unit)) {
//	c.doAllWeights(func(tag string, ids []int, u *Unit) {
//		s := tag
//		for i := len(ids) - 1; i >= 0; i-- {
//			s = fmt.Sprintf("%s[%d]", s, ids[i])
//		}
//		f(s, u)
//	})
//}
//
//func (c *Controller2) ClearGradients() {
//	c.doAllWeights(func(tag string, ids []int, u *Unit) { u.Grad = 0 })
//}
//
//func (c *Controller2) NumWeights() int {
//	return c.numWeights
//}
//
//func (c *Controller2) NumHeads() int {
//	return len(c.WUh1)
//}
//
//func (c *Controller2) MemoryM() int {
//	return len(c.WUh1r[0][0])
//}
//
//func (c *Controller2) doAllWeights(f func(tag string, ids []int, u *Unit)) {
//	doUnit3(c.WYh1r, func(ids []int, u *Unit) { f("WYh1r", ids, u) })
//	doUnit2(c.WYh1x, func(ids []int, u *Unit) { f("WYh1x", ids, u) })
//	doUnit2(c.WYh1, func(ids []int, u *Unit) { f("WYh1", ids, u) })
//	doUnit3(c.WUh1r, func(ids []int, u *Unit) { f("WUh1r", ids, u) })
//	doUnit2(c.WUh1x, func(ids []int, u *Unit) { f("WUh1x", ids, u) })
//	doUnit3(c.WUh1, func(ids []int, u *Unit) { f("WUh1", ids, u) })
//}
