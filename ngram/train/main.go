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

	"github.com/fumin/ntm"
	"github.com/fumin/ntm/ngram"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	weightsChan    = make(chan chan []byte)
	lossChan       = make(chan chan []float64)
	printDebugChan = make(chan struct{})
)

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
	port := 8087
	go func() {
		log.Printf("Listening on port %d", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
			log.Fatalf("%v", err)
		}
	}()

	var seed int64 = 7
	rand.Seed(seed)

	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	c := ntm.NewEmptyController1(1, 1, h1Size, numHeads, n, m)
	weights := c.WeightsVal()
	for i := range weights {
		weights[i] = 1 * (rand.Float64() - 0.5)
	}

	losses := make([]float64, 0)
	doPrint := false

	rmsp := ntm.NewRMSProp(c)
	log.Printf("seed: %d, numweights: %d, numHeads: %d", seed, len(c.WeightsVal()), c.NumHeads())
	for i := 1; ; i++ {
		x, y := ngram.GenSeq(ngram.GenProb())
		machines := rmsp.Train(x, &ntm.LogisticModel{Y: y}, 0.95, 0.5, 1e-3, 1e-3)

		if i%1000 == 0 {
			prob := ngram.GenProb()
			var l float64 = 0
			samn := 100
			for j := 0; j < samn; j++ {
				x, y = ngram.GenSeq(prob)
				model := &ntm.LogisticModel{Y: y}
				machines = ntm.ForwardBackward(c, x, model)
				l += model.Loss(ntm.Predictions(machines))
			}
			l = l / float64(samn)
			losses = append(losses, l)
			log.Printf("%d, bits-per-seq: %f", i, l)
		}

		handleHTTP(c, losses, &doPrint)

		if i%1000 == 0 && doPrint {
			printDebug(x, y, machines)
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

func printDebug(x, y [][]float64, machines []*ntm.NTM) {
	log.Printf("x: %+v", x)
	log.Printf("y: %+v", y)
	log.Printf("pred: %s", ntm.Sprint2(ntm.Predictions(machines)))
}
