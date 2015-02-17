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
	port := 8088
	go func() {
		log.Printf("Listening on port %d", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
			log.Fatalf("%v", err)
		}
	}()

	var seed int64 = 8
	rand.Seed(seed)
	log.Printf("seed: %d", seed)

	vectorSize := 8
	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	c := ntm.NewEmptyController1(vectorSize+2, vectorSize, h1Size, numHeads, n, m)
	c.Weights(func(u *ntm.Unit) { u.Val = 1 * (rand.Float64() - 0.5) })

	losses := make([]float64, 0)
	doPrint := false

	//sgd := ntm.NewSGDMomentum(c)
	rmsp := ntm.NewRMSProp(c)
	log.Printf("numweights: %d", c.NumWeights())
	for i := 1; ; i++ {
		x, y := copytask.GenSeq(rand.Intn(20)+1, vectorSize)
		//machines := sgd.Train(x, y, 1e-4, 0.9)
		machines := rmsp.Train(x, y, 0.95, 0.5, 1e-3, 1e-3)
		l := ntm.Loss(y, machines)
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
		ws := make([]float64, 0, c.NumWeights())
		c.Weights(func(u *ntm.Unit) { ws = append(ws, u.Val) })
		b, err := json.Marshal(ws)
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

	n := len(machines[0].Circuit.WM.Top)
	//outputT := len(machines) - (len(machines) - 2) / 2
	outputT := 0
	for t := outputT; t < len(machines); t++ {
		h := machines[t].Controller.Heads()[0]
		beta := math.Exp(h.Beta().Val)
		g := ntm.Sigmoid(h.G().Val)
		shift := math.Mod(2*ntm.Sigmoid(h.S().Val)-1+float64(n), float64(n))
		gamma := math.Log(math.Exp(h.Gamma().Val)+1) + 1
		log.Printf("beta: %.3g(%v), g: %.3g(%v), s: %.3g(%v), gamma: %.3g(%v), erase: %+v, add: %+v, k: %+v", beta, h.Beta(), g, h.G(), shift, h.S(), gamma, h.Gamma(), h.EraseVector(), h.AddVector(), h.K())
	}
}
