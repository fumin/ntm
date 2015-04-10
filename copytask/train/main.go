package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime/pprof"

	"github.com/fumin/ntm"
	"github.com/fumin/ntm/copytask"
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
	port := 8082
	go func() {
		log.Printf("Listening on port %d", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
			log.Fatalf("%v", err)
		}
	}()

	var seed int64 = 2
	rand.Seed(seed)
	log.Printf("seed: %d", seed)

	vectorSize := 8
	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	c := ntm.NewEmptyController1(vectorSize+2, vectorSize, h1Size, numHeads, n, m)
	weights := c.WeightsVal()
	for i := range weights {
		weights[i] = 1 * (rand.Float64() - 0.5)
	}

	losses := make([]float64, 0)
	doPrint := false

	//sgd := ntm.NewSGDMomentum(c)
	rmsp := ntm.NewRMSProp(c)
	log.Printf("numweights: %d", len(c.WeightsVal()))
	for i := 1; ; i++ {
		x, y := copytask.GenSeq(rand.Intn(20)+1, vectorSize)
		model := &ntm.LogisticModel{Y: y}
		//machines := sgd.Train(x, model, 1e-4, 0.9)
		machines := rmsp.Train(x, model, 0.95, 0.5, 1e-3, 1e-3)
		l := model.Loss(ntm.Predictions(machines))
		if i%1000 == 0 {
			bpc := l / float64(len(y)*len(y[0]))
			losses = append(losses, bpc)
			log.Printf("%d, bpc: %f, seq length: %d", i, bpc, len(y))
		}

		handleHTTP(c, losses, &doPrint)

		if i%1000 == 0 && doPrint {
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

func printDebug(y [][]float64, machines []*ntm.NTM) {
	log.Printf("y: %+v", y)

	log.Printf("pred: %s", ntm.Sprint2(ntm.Predictions(machines)))

	n := machines[0].Controller.MemoryN()
	//outputT := len(machines) - (len(machines) - 2) / 2
	outputT := 0
	for t := outputT; t < len(machines); t++ {
		h := machines[t].Controller.Heads()[0]
		beta := math.Exp(*h.BetaVal())
		g := ntm.Sigmoid(*h.GVal())
		shift := math.Mod(2*ntm.Sigmoid(*h.SVal())-1+float64(n), float64(n))
		gamma := math.Log(math.Exp(*h.GammaVal())+1) + 1
		log.Printf("beta: %.3g(%v), g: %.3g(%v), s: %.3g(%v), gamma: %.3g(%v), erase: %+v, add: %+v, k: %+v", beta, *h.BetaVal(), g, *h.GVal(), shift, *h.SVal(), gamma, *h.GammaVal(), h.EraseVal(), h.AddVal(), h.KVal())
	}
}
