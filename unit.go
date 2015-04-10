package ntm

import (
	"fmt"
)

// A Unit is a node in a neural network, containing fields that are essential to efficiently compute gradients in the backward pass of a stochastic gradient descent training process.
type Unit struct {
	Val  float64 // value at node
	Grad float64 // gradient at node
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

func doUnit1(t []Unit, f func(*Unit)) {
	for i := range t {
		f(&t[i])
	}
}

func doUnit2(t [][]Unit, f func(*Unit)) {
	for _, v := range t {
		doUnit1(v, f)
	}
}

func doUnit3(t [][][]Unit, f func(*Unit)) {
	for _, v := range t {
		doUnit2(v, f)
	}
}

func doUnit1Indices(t []Unit, f func([]int, *Unit)) {
	for i := 0; i < len(t); i++ {
		f([]int{i}, &t[i])
	}
}

func doUnit2Indices(t [][]Unit, f func([]int, *Unit)) {
	for i, a := range t {
		doUnit1Indices(a, func(ids []int, u *Unit) { f(append(ids, i), u) })
	}
}

func doUnit3Indices(t [][][]Unit, f func([]int, *Unit)) {
	for i, a := range t {
		doUnit2Indices(a, func(ids []int, u *Unit) { f(append(ids, i), u) })
	}
}
