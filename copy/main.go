package main

import (
	"flag"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/fumin/ntm"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
)

func genSeq(size, vectorSize int) ([][]float64, [][]float64) {
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

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	vectorSize := 8
	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	w := ntm.NewControllerWs(vectorSize+2, vectorSize, h1Size, numHeads, m)
	// Weights cannot be zero, or else we have division by zero in cosine similarity of content addressing.
	ntm.RandVal3(w.Wh1r)
	ntm.RandVal2(w.Wh1x)
	ntm.RandVal2(w.Wyh1)
	ntm.RandVal3(w.Wuh1)

	sgd := ntm.NewSGDMomentum(w)
	alpha := 0.0001
	momentum := 0.9
	for i := 1; ; i++ {
		x, y := genSeq(rand.Intn(20)+1, vectorSize)
		machines := sgd.Train(x, y, n, alpha, momentum)
		l := loss(y, machines)
		if i%1000 == 0 {
			log.Printf("%d, bits-per-sequence: %f", i, l/float64(len(y)*len(y[0])))
		}
	}
}

func loss(output [][]float64, ms []*ntm.NTM) float64 {
	var l float64 = 0
	for t := 0; t < len(output); t++ {
		for i := 0; i < len(output[t]); i++ {
			y := output[t][i]
			p := ms[t].Controller.Y[i].Val
			l += y*math.Log2(p) + (1-y)*math.Log2(1-p)
		}
	}
	return -l
}
