package ntm

import (
	"fmt"
)

type Unit struct {
	Val  float64
	Grad float64
}

func (u Unit) String() string {
	return fmt.Sprintf("{%.3g %.3g}", u.Val, u.Grad)
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

func doUnit1(t []Unit, f func([]int, *Unit)) {
	for i := 0; i < len(t); i++ {
		f([]int{i}, &t[i])
	}
}

func doUnit2(t [][]Unit, f func([]int, *Unit)) {
	for i, a := range t {
		doUnit1(a, func(ids []int, u *Unit) { f(append(ids, i), u) })
	}
}

func doUnit3(t [][][]Unit, f func([]int, *Unit)) {
	for i, a := range t {
		doUnit2(a, func(ids []int, u *Unit) { f(append(ids, i), u) })
	}
}
