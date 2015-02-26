package ngram

import (
	"math"
	"math/rand"
)

// GenProb generates a probability lookup table for a n-gram model.
func GenProb() []float64 {
	n := 5
	probs := make([]float64, 1<<uint(n))
	for i := range probs {
		probs[i] = beta()
	}
	return probs
}

func GenSeq(prob []float64) ([][]float64, [][]float64) {
	n := int(math.Log2(float64(len(prob))))
	seqLen := 200

	input := make([][]float64, seqLen+1)
	for i := 0; i < n; i++ {
		input[i] = []float64{float64(rand.Intn(2))}
	}
	for i := n; i < len(input); i++ {
		idx := Binarize(input[i-n : i])
		if rand.Float64() < prob[idx] {
			input[i] = []float64{1}
		} else {
			input[i] = []float64{0}
		}
	}

	output := make([][]float64, seqLen)
	for i := 0; i < n-1; i++ {
		output[i] = []float64{0}
	}
	copy(output[n-1:], input[n:])
	return input[0:seqLen], output
}

func Binarize(seq [][]float64) int {
	idx := 0
	for i, a := range seq {
		idx += int(a[0]) * (1 << uint(i))
	}
	return idx
}

// beta generates a random number from the Beta(1/2, 1/2) distribution.
func beta() float64 {
	x := gamma()
	y := gamma()
	return x / (x + y)
}

// gamma generates a random number from the Gamma(1/2, 1) distribution.
func gamma() float64 {
	n := rand.NormFloat64()
	return 0.5 * n * n
}
