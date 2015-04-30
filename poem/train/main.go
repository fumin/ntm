package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"runtime/pprof"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"

	"github.com/fumin/ntm"
	"github.com/fumin/ntm/poem"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	weightsChan    = make(chan chan []byte)
	lossChan       = make(chan chan []float64)
	printDebugChan = make(chan struct{})
)

func main() {
	flag.Parse()
	blas64.Use(cgo.Implementation{})

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	http.HandleFunc("/Weights", func(w http.ResponseWriter, r *http.Request) {
		c := make(chan []byte)
		weightsChan <- c
		w.Write(<-c)
	})
	http.HandleFunc("/Loss", func(w http.ResponseWriter, r *http.Request) {
		c := make(chan []float64)
		lossChan <- c
		json.NewEncoder(w).Encode(<-c)
	})
	http.HandleFunc("/PrintDebug", func(w http.ResponseWriter, r *http.Request) {
		printDebugChan <- struct{}{}
	})
	port := 8085
	go func() {
		log.Printf("Listening on port %d", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
			log.Fatalf("%v", err)
		}
	}()

	var seed int64 = 5
	rand.Seed(seed)
	log.Printf("seed: %d", seed)

	gen, err := poem.NewGenerator("data/quantangshi3000.int")
	if err != nil {
		log.Fatalf("%v", err)
	}
	h1Size := 512
	numHeads := 8
	n := 128
	m := 32
	c := ntm.NewEmptyController1(gen.InputSize(), gen.OutputSize(), h1Size, numHeads, n, m)
	weights := c.WeightsVal()
	for i := range weights {
		weights[i] = 1 * (rand.Float64() - 0.5)
	}

	losses := make([]float64, 0)
	doPrint := false

	rmsp := ntm.NewRMSProp(c)
	log.Printf("numweights: %d", len(c.WeightsVal()))
	var bpcSum float64 = 0
	for i := 1; ; i++ {
		x, y := gen.GenSeq()
		machines := rmsp.Train(x, &ntm.MultinomialModel{Y: y}, 0.95, 0.5, 1e-3, 1e-3)

		numChar := len(y) / 2
		l := (&ntm.MultinomialModel{Y: y[numChar+1:]}).Loss(ntm.Predictions(machines[numChar+1:]))
		bpc := l / float64(numChar)
		bpcSum += bpc

		acc := 100
		if i%acc == 0 {
			bpc := bpcSum / float64(acc)
			bpcSum = 0
			losses = append(losses, bpc)
			log.Printf("%d, bpc: %f, seq length: %d", i, bpc, len(y))
		}

		handleHTTP(c, losses, &doPrint)

		if i%10 == 0 && doPrint {
			printDebug(y, machines)
		}
	}
}

func handleHTTP(c ntm.Controller, losses []float64, doPrint *bool) {
	select {
	case cn := <-weightsChan:
		b, err := json.Marshal(c.WeightsVal())
		if err != nil {
			log.Fatalf("%v", err)
		}
		cn <- b
	case cn := <-lossChan:
		cn <- losses
	case <-printDebugChan:
		*doPrint = !*doPrint
	default:
		return
	}
}

func printDebug(y []int, machines []*ntm.NTM) {
	log.Printf("y: %+v", y)

	log.Printf("pred: %s", ntm.Sprint2(ntm.Predictions(machines)))
}
