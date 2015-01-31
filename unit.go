package ntm

import (
	"math/rand"
)

type Unit struct {
	Val  float64
	Grad float64
}

func makeTensorUnit2(n, m int) [][]Unit {
	t := make([][]Unit, n)
	for i := 0; i < len(t); i++ {
		t[i] = make([]Unit, m)
	}
	return t
}

func makeTensorUnit3(n, m, p int) [][][]Unit {
	t := make([][][]Unit, n)
	for i := 0; i < len(t); i++ {
		t[i] = makeTensorUnit2(m, p)
	}
	return t
}

func unitVals(units []Unit) []float64 {
	v := make([]float64, 0, len(units))
	for _, u := range units {
		v = append(v, u.Val)
	}
	return v
}

func clearGrad1(t []Unit) {
	for i := 0; i < len(t); i++ {
		t[i].Grad = 0
	}
}

func clearGrad2(t [][]Unit) {
	for i := 0; i < len(t); i++ {
		clearGrad1(t[i])
	}
}

func clearGrad3(t [][][]Unit) {
	for i := 0; i < len(t); i++ {
		clearGrad2(t[i])
	}
}

func RandVal1(t []Unit) {
	for i := 0; i < len(t); i++ {
		t[i].Val = rand.NormFloat64()
	}
}

func RandVal2(t [][]Unit) {
	for i := 0; i < len(t); i++ {
		RandVal1(t[i])
	}
}

func RandVal3(t [][][]Unit) {
	for i := 0; i < len(t); i++ {
		RandVal2(t[i])
	}
}
