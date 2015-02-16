package ntm

import (
	"math"
)

const (
	machineEpsilon     = 2.2e-16
	machineEpsilonSqrt = 1e-8 // math.Sqrt(machineEpsilon)
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

func similarity(u, v []float64) float64 {
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

func MakeTensor2(n, m int) [][]float64 {
	t := make([][]float64, n)
	for i := 0; i < len(t); i++ {
		t[i] = make([]float64, m)
	}
	return t
}

func MakeTensor3(n, m, p int) [][][]float64 {
	t := make([][][]float64, n)
	for i := 0; i < len(t); i++ {
		t[i] = MakeTensor2(m, p)
	}
	return t
}
