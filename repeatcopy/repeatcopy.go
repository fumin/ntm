package repeatcopy

import (
	"math/rand"
	"strconv"
)

type genFunc func(int, int) ([][]float64, [][]float64)

var (
	G = map[string]genFunc{
		"bt": GenSeqBT,
		"lt": GenSeqLT,
		"":   GenSeq,
	}
)

// binary on time
func GenSeqBT(repeat, seqlen int) ([][]float64, [][]float64) {
	data := randData(seqlen)
	vectorSize := len(data[0])
	inputSize := vectorSize + 4
	outputSize := vectorSize + 1

	input := make([][]float64, 0)
	marker := make([]float64, inputSize)
	marker[vectorSize] = 1
	input = append(input, marker)

	for _, datum := range data {
		v := make([]float64, inputSize)
		copy(v, datum)
		input = append(input, v)
	}

	marker = make([]float64, inputSize)
	marker[vectorSize+1] = 1
	input = append(input, marker)

	// Encode repeat times as little endian.
	repeatBin := strconv.FormatInt(int64(repeat), 2)
	for i := len(repeatBin) - 1; i >= 0; i-- {
		v := make([]float64, inputSize)
		if repeatBin[i] == '1' {
			v[vectorSize+2] = 1
		} else {
			v[vectorSize+2] = 0
		}
		input = append(input, v)
	}

	v := make([]float64, inputSize)
	v[vectorSize+3] = 1
	input = append(input, v)

	output := make([][]float64, len(input))
	for i := 0; i < len(input); i++ {
		output[i] = make([]float64, outputSize)
	}
	for i := 0; i < repeat; i++ {
		for _, datum := range data {
			input = append(input, make([]float64, inputSize))
			v := make([]float64, outputSize)
			copy(v, datum)
			output = append(output, v)
		}
	}

	input = append(input, make([]float64, inputSize))
	marker = make([]float64, outputSize)
	marker[vectorSize] = 1
	output = append(output, marker)

	return input, output
}

// linear on time
func GenSeqLT(repeat, seqlen int) ([][]float64, [][]float64) {
	data := randData(seqlen)
	vectorSize := len(data[0])
	inputSize := vectorSize + 3
	outputSize := vectorSize + 1

	input := make([][]float64, 0)
	marker := make([]float64, inputSize)
	marker[vectorSize] = 1
	input = append(input, marker)

	for _, datum := range data {
		v := make([]float64, inputSize)
		copy(v, datum)
		input = append(input, v)
	}

	// Encode repeat times as repititions.
	for i := 0; i < repeat; i++ {
		v := make([]float64, inputSize)
		v[vectorSize+1] = 1
		input = append(input, v)
	}

	v := make([]float64, inputSize)
	v[vectorSize+2] = 1
	input = append(input, v)

	output := make([][]float64, len(input))
	for i := 0; i < len(input); i++ {
		output[i] = make([]float64, outputSize)
	}
	for i := 0; i < repeat; i++ {
		for _, datum := range data {
			input = append(input, make([]float64, inputSize))
			v := make([]float64, outputSize)
			copy(v, datum)
			output = append(output, v)
		}
	}

	input = append(input, make([]float64, inputSize))
	marker = make([]float64, outputSize)
	marker[vectorSize] = 1
	output = append(output, marker)

	return input, output
}

// GenSeq generates a sequence with the number of repitions specified as a scaler.
func GenSeq(repeat, seqlen int) ([][]float64, [][]float64) {
	data := randData(seqlen)
	vectorSize := len(data[0])
	inputSize := vectorSize + 2
	outputSize := vectorSize + 1

	input := make([][]float64, 0)
	marker := make([]float64, inputSize)
	marker[vectorSize] = 1
	input = append(input, marker)

	for _, datum := range data {
		v := make([]float64, inputSize)
		copy(v, datum)
		input = append(input, v)
	}

	// Encode repeat times as a scalar.
	v := make([]float64, inputSize)
	v[vectorSize+1] = float64(repeat)
	input = append(input, v)

	output := make([][]float64, len(input))
	for i := 0; i < len(input); i++ {
		output[i] = make([]float64, outputSize)
	}
	for i := 0; i < repeat; i++ {
		for _, datum := range data {
			input = append(input, make([]float64, inputSize))
			v := make([]float64, outputSize)
			copy(v, datum)
			output = append(output, v)
		}
	}

	input = append(input, make([]float64, inputSize))
	marker = make([]float64, outputSize)
	marker[vectorSize] = 1
	output = append(output, marker)

	return input, output
}

func randData(size int) [][]float64 {
	vectorSize := 6
	data := make([][]float64, size)
	for i := 0; i < len(data); i++ {
		data[i] = make([]float64, vectorSize)
		for j := 0; j < len(data[i]); j++ {
			data[i][j] = float64(rand.Intn(2))
		}
	}
	return data
}
