package main

import (
	"encoding/json"
	"flag"
	"html/template"
	"log"
	"net/http"
	"os"

	"github.com/fumin/ntm"
	"github.com/fumin/ntm/ngram"
)

var (
	weightsFile = flag.String("weightsFile", "", "trained weights in JSON")
)

type Run struct {
	Conf        RunConf
	BitsPerSeq  float64
	X           [][]float64
	Y           [][]float64
	Predictions [][]float64
	HeadWeights [][][]float64
}

type RunConf struct {
	Prob []float64
}

func main() {
	flag.Parse()

	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	c := ntm.NewEmptyController1(1, 1, h1Size, numHeads, n, m)
	weightsFromFile(c)

	runs := make([]Run, 0)
	for i := 0; i < 1; i++ {
		prob := ngram.GenProb()
		var l float64 = 0
		var x [][]float64
		var y [][]float64
		var machines []*ntm.NTM
		sampletimes := 100
		for j := 0; j < sampletimes; j++ {
			x, y = ngram.GenSeq(prob)
			model := &ntm.LogisticModel{Y: y}
			machines = ntm.ForwardBackward(c, x, model)
			l += model.Loss(ntm.Predictions(machines))
			if (j+1)%10 == 0 {
				log.Printf("%d %d %f", i, j+1, l/float64(j+1))
			}
		}
		l = l / float64(sampletimes)

		r := Run{
			Conf:        RunConf{Prob: prob},
			BitsPerSeq:  l,
			X:           x,
			Y:           y,
			Predictions: ntm.Predictions(machines),
			HeadWeights: ntm.HeadWeights(machines),
		}
		runs = append(runs, r)
		//log.Printf("x: %v", x)
		//log.Printf("y: %v", y)
		//log.Printf("predictions: %s", ntm.Sprint2(ntm.Predictions(machines)))
	}

	http.HandleFunc("/", root(runs))
	if err := http.ListenAndServe(":9000", nil); err != nil {
		log.Printf("%v", err)
	}
}

var rootTmpl = template.Must(template.New("").Parse(`
<!DOCTYPE html>
<html>
<head>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
</head>
<body>
<script type="text/javascript">
var page = {{.}};

var colorbrewer = {};
colorbrewer.Greys = {};
colorbrewer.Greys[9] = ["#ffffff","#f0f0f0","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525","#000000"];

// palette draws a color palette explaining that 0.0 maps to blue and 1.0 maps to red.
function palette(parent) {
  var colormap = colorbrewer.Greys[9];
  var matrix = colormap.map(function(d, i) {
    return [{"text": ""}, {"bgcolor": d}];
  });
  matrix[0][0].text = "1.0";
  matrix[(colormap.length-1) / 2][0].text = "0.5";
  matrix[colormap.length-1][0].text = "0.0";
  var table = parent.append("table")
  var tr = table.selectAll("tr").data(matrix).
    enter().append("tr");
  var td = tr.selectAll("td").data(function(d) { return d; }).
    enter().append("td").
    text(function(d) { return d.text; }).
    style("background-color", function(d) { return d.bgcolor; }).
    style("min-width", "1em").
    style("height", "1em");
  return table;
}

// imshow displays a 2 dimensional matrix.
function imshow(parent, matrix) {
  var table = parent.append("table");

  var tr = table.selectAll("tr").data(matrix).
    enter().append("tr");

  var colormap = d3.scale.quantize().domain([0, 1]).range(colorbrewer.Greys[9].slice().reverse());
  var td = tr.selectAll("td").data(function(d) { return d; }).
    enter().append("td").
    style("background-color", colormap).
    style("min-width", "1em").
    style("height", "1em");
  return table;
}

var allRuns = d3.select("body").append("div").attr("id", "runs");
var run = allRuns.selectAll("div").
  data(page.Runs).
  enter().append("div").
  attr("id", function(d){ return "run-"+d.SeqLen;});

run.append("h4").text(function(d){ return "bits-per-sequence: "+d.BitsPerSeq.toPrecision(3); });

// Draw x along with a palette.
run.append("h5").text("input:");
var x = run.append("table").style("border-spacing", "0px").append("tr");
imshow(x.append("td").style("padding-left", "0px"), function(d){ return d3.transpose(d.X); });
palette(x.append("td"));

run.append("h5").text("output and prediction:");
imshow(run, function(d){ return d3.transpose(d.Y); });
imshow(run, function(d){ return d3.transpose(d.Predictions); });

// Draw the weights of all memory heads.
var headWs = run.append("div");
headWs.append("h5").text("Head weights:");
headWs.selectAll("div").
  data(function(d){ return d.HeadWeights; }).
  enter().call(imshow, function(d){ return d3.transpose(d); });
</script>
<body>
</html>
`))

func root(runs []Run) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		page := struct {
			Runs []Run
		}{
			Runs: runs,
		}
		rootTmpl.Execute(w, page)
	}
}

func weightsFromFile(c ntm.Controller) {
	if *weightsFile == "" {
		flag.PrintDefaults()
		os.Exit(1)
	}

	f, err := os.Open(*weightsFile)
	if err != nil {
		log.Fatalf("%v", err)
	}
	defer f.Close()
	ws := c.WeightsVal()
	if err := json.NewDecoder(f).Decode(&ws); err != nil {
		log.Fatalf("%v", err)
	}
}
