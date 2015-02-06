package ntm

//
//import (
//	"math/rand"
//	"testing"
//)
//
//func TestController2(t *testing.T) {
//	times := 40
//	x := MakeTensor2(times, 2)
//	for i := 0; i < len(x); i++ {
//		for j := 0; j < len(x[i]); j++ {
//			x[i][j] = rand.Float64()
//		}
//	}
//	y := MakeTensor2(times, 2)
//	for i := 0; i < len(y); i++ {
//		for j := 0; j < len(y[i]); j++ {
//			y[i][j] = rand.Float64()
//		}
//	}
//	n := 3
//	m := 2
//	h1Size := 3
//	numHeads := 2
//	c := NewEmptyController2(len(x[0]), len(y[0]), h1Size, numHeads, m)
//	c.Weights(func(tag string, u *Unit) { u.Val = rand.Float64() })
//	forwardBackward(c, n, x, y)
//
//	l := loss(c, Controller2Forward, n, x, y)
//	checkGradients(t, c, Controller2Forward, n, x, y, l)
//}
//
//func Controller2Forward(c2 Controller, reads [][]float64, x []float64) ([]float64, []*Head) {
//	c := c2.(*Controller2)
//	yh1 := make([]float64, len(c.WYh1r))
//	for i := 0; i < len(yh1); i++ {
//		var v float64 = 0
//		for j := 0; j < len(reads); j++ {
//			for k := 0; k < len(reads[j]); k++ {
//				v += c.WYh1r[i][j][k].Val * reads[j][k]
//			}
//		}
//		for j := 0; j < len(x); j++ {
//			v += c.WYh1x[i][j].Val * x[j]
//		}
//		yh1[i] = sigmoid(v)
//	}
//	prediction := make([]float64, len(c.WYh1))
//	for i := 0; i < len(prediction); i++ {
//		var v float64 = 0
//		for j := 0; j < len(yh1); j++ {
//			v += c.WYh1[i][j].Val * yh1[j]
//		}
//		prediction[i] = sigmoid(v)
//	}
//
//	uh1 := make([]float64, len(c.WUh1r))
//	for i := 0; i < len(uh1); i++ {
//		var v float64 = 0
//		for j := 0; j < len(reads); j++ {
//			for k := 0; k < len(reads[j]); k++ {
//				v += c.WUh1r[i][j][k].Val * reads[j][k]
//			}
//		}
//		for j := 0; j < len(x); j++ {
//			v += c.WUh1x[i][j].Val * x[j]
//		}
//		uh1[i] = sigmoid(v)
//	}
//	heads := make([]*Head, len(reads))
//	m := len(reads[0])
//	for i := 0; i < len(heads); i++ {
//		heads[i] = NewHead(m)
//		for j := 0; j < len(heads[i].units); j++ {
//			for k := 0; k < len(c.WUh1[i][j]); k++ {
//				heads[i].units[j].Val += c.WUh1[i][j][k].Val * uh1[k]
//			}
//		}
//	}
//	return prediction, heads
//}
