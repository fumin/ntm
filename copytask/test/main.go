package main

import (
	"encoding/json"
	"flag"
	"html/template"
	"log"
	"net/http"
	"os"

	"github.com/fumin/ntm"
	"github.com/fumin/ntm/copytask"
)

var (
	weightsFile = flag.String("weightsFile", "", "trained weights in JSON")
)

type Run struct {
	SeqLen      int
	BitsPerSeq  float64
	X           [][]float64
	Y           [][]float64
	Predictions [][]float64
	HeadWeights [][][]float64
}

func main() {
	flag.Parse()
	vectorSize := 8
	h1Size := 100
	numHeads := 1
	n := 128
	m := 20
	c := ntm.NewEmptyController1(vectorSize+2, vectorSize, h1Size, numHeads, n, m)
	copy(c.WeightsVal(), weightsFromFile())

	seqLens := []int{10, 20, 30, 50, 120}
	runs := make([]Run, 0, len(seqLens))
	for _, seql := range seqLens {
		x, y := copytask.GenSeq(seql, vectorSize)
		model := &ntm.LogisticModel{Y: y}
		machines := ntm.ForwardBackward(c, x, model)
		l := model.Loss(ntm.Predictions(machines))
		bps := l / float64(len(y)*len(y[0]))
		log.Printf("sequence length: %d, loss: %f", seql, bps)

		r := Run{
			SeqLen:      seql,
			BitsPerSeq:  bps,
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
  <script type="text/javascript" src="http://d3js.org/d3.v3.js"></script>
</head>
<body>
<script type="text/javascript">
var page = {{.}};

var colorbrewer = {};
colorbrewer.RdYlBu = {};
colorbrewer.RdYlBu[9] = ["#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4"];

// palette draws a color palette explaining that 0.0 maps to blue and 1.0 maps to red.
function palette(parent) {
  var matrix = colorbrewer.RdYlBu[9].map(function(d, i) {
    return [{"text": ""}, {"bgcolor": d}];
  });
  matrix[0][0].text = "1.0";
  matrix[(colorbrewer.RdYlBu[9].length-1) / 2][0].text = "0.5";
  matrix[colorbrewer.RdYlBu[9].length-1][0].text = "0.0";
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
  var colormap = d3.scale.quantize().domain([0, 1]).range(colorbrewer.RdYlBu[9].slice().reverse());
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

run.append("h4").text(function(d){ return "Sequence length: "+d.SeqLen+", bits-per-char: "+d.BitsPerSeq.toPrecision(3); });

// Draw x along with a palette.
var x = run.append("table").style("border-spacing", "0px").append("tr");
imshow(x.append("td").style("padding-left", "0px"), function(d){ return d3.transpose(d.X); });
palette(x.append("td"));

// Draw predictions
imshow(run, function(d){ return d3.transpose(d.Y); });
imshow(run, function(d){ return d3.transpose(d.Predictions); });

var headWs = run.append("div");
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

func weightsFromFile() []float64 {
	if *weightsFile == "" {
		flag.PrintDefaults()
		os.Exit(1)
	}

	f, err := os.Open(*weightsFile)
	if err != nil {
		log.Fatalf("%v", err)
	}
	defer f.Close()
	ws := make([]float64, 0)
	if err := json.NewDecoder(f).Decode(&ws); err != nil {
		log.Fatalf("%v", err)
	}
	return ws
}
