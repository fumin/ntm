package ntm

import (
	"fmt"
	"math"
)

const (
	machineEpsilon     = 2.2e-16
	machineEpsilonSqrt = 1e-8 // math.Sqrt(machineEpsilon)
)

// Sigmoid computes 1 / (1 + math.Exp(-x))
func Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

func delta(a, b int) float64 {
	if a == b {
		return 1
	}
	return 0
}

func Mul(dst, s []float64) {
	for i, val := range s {
		dst[i] *= val
	}
}

func cosineSimilarity(u, v []float64) float64 {
	var sum float64 = 0
	var usum float64 = 0
	var vsum float64 = 0
	for i := 0; i < len(u); i++ {
		sum += u[i] * v[i]
		usum += u[i] * u[i]
		vsum += v[i] * v[i]
	}
	return sum / math.Sqrt(usum*vsum)
}

// MakeTensor2 makes a 2 dimensional tensor.
func MakeTensor2(n, m int) [][]float64 {
	t := make([][]float64, n)
	for i := 0; i < len(t); i++ {
		t[i] = make([]float64, m)
	}
	return t
}

// MakeTensor3 makes a 3 dimensional tensor.
func MakeTensor3(n, m, p int) [][][]float64 {
	t := make([][][]float64, n)
	for i := 0; i < len(t); i++ {
		t[i] = MakeTensor2(m, p)
	}
	return t
}

// Sprint2 pretty prints a 2 dimensional tensor.
func Sprint2(t [][]float64) string {
	s := "["
	for _, t1 := range t {
		s += "["
		for _, t2 := range t1 {
			s += fmt.Sprintf(" %.2f", t2)
		}
		s += "]"
	}
	s += "]"
	return s
}
