package copytask

import (
	"math/rand"
)

func GenSeq(size, vectorSize int) ([][]float64, [][]float64) {
	data := make([][]float64, size)
	for i := 0; i < len(data); i++ {
		data[i] = make([]float64, vectorSize)
		for j := 0; j < len(data[i]); j++ {
			data[i][j] = float64(rand.Intn(2))
		}
	}

	input := make([][]float64, size*2+2)
	for i := 0; i < len(input); i++ {
		input[i] = make([]float64, vectorSize+2)
		if i == 0 {
			input[i][vectorSize] = 1
		} else if i <= size {
			for j := 0; j < vectorSize; j++ {
				input[i][j] = data[i-1][j]
			}
		} else if i == size+1 {
			input[i][vectorSize+1] = 1
		}
	}

	output := make([][]float64, size*2+2)
	for i := 0; i < len(output); i++ {
		output[i] = make([]float64, vectorSize)
		if i >= size+2 {
			for j := 0; j < vectorSize; j++ {
				output[i][j] = data[i-(size+2)][j]
			}
		}
	}

	return input, output
}
