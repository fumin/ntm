package poem

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
)

const (
	indexLinefeed  = 1
	indexEndOfPoem = 2
)

type Dataset struct {
	Chars map[string]int
	Shis  [][][]int
}

type Generator struct {
	Dataset     Dataset
	IndexToChar map[int]string

	indices []int
	offset  int
}

func NewGenerator(filepath string) (*Generator, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g := Generator{IndexToChar: make(map[int]string)}
	if err := json.NewDecoder(f).Decode(&g.Dataset); err != nil {
		return nil, err
	}

	for s, i := range g.Dataset.Chars {
		g.IndexToChar[i] = s
	}

	g.indices = make([]int, len(g.Dataset.Shis))
	g.resample()
	return &g, nil
}

func (g *Generator) GenSeq() ([][]float64, []int) {
	poem := g.Dataset.Shis[g.indices[g.offset]]
	g.offset += 1
	if g.offset == len(g.indices) {
		g.resample()
	}

	// Limit poem size to avoid memory issues.
	// For poems with lines over 200, we might need over 10GB of RAM.
	if len(poem) > 32 {
		poem = poem[0:32]
	}

	inputSize := g.InputSize()
	input := make([][]float64, 0)
	for _, line := range poem {
		jndex := rand.Intn(len(line))
		for j, c := range line {
			cv := make([]float64, inputSize)
			if j == jndex {
				cv[c] = 1
			}
			input = append(input, cv)
		}
		input = append(input, g.Linefeed())
	}
	input = append(input, g.EndOfPoem())

	output := make([]int, 0)
	for range input {
		output = append(output, 0)
	}

	prevC := -1
	for _, line := range poem {
		for _, c := range line {
			cvi := make([]float64, inputSize)
			if prevC >= 0 {
				cvi[prevC] = 1
			}
			input = append(input, cvi)
			prevC = c

			output = append(output, c)
		}

		cvi := make([]float64, inputSize)
		cvi[prevC] = 1
		input = append(input, cvi)
		prevC = len(g.Dataset.Chars) + indexLinefeed

		output = append(output, len(g.Dataset.Chars)+indexLinefeed)
	}

	return input, output
}

func (g *Generator) Linefeed() []float64 {
	v := make([]float64, g.InputSize())
	v[len(g.Dataset.Chars)+indexLinefeed] = 1
	return v
}

func (g *Generator) EndOfPoem() []float64 {
	v := make([]float64, g.InputSize())
	v[len(g.Dataset.Chars)+indexEndOfPoem] = 1
	return v
}

func (g *Generator) InputSize() int {
	return len(g.Dataset.Chars) + indexEndOfPoem + 1
}

func (g *Generator) OutputSize() int {
	return len(g.Dataset.Chars) + indexLinefeed + 1
}

func (g *Generator) resample() {
	g.indices = rand.Perm(len(g.indices))
	g.offset = 0
}

func (g *Generator) SortOutput(output []float64) []Char {
	chars := make([]Char, len(output))
	for i, o := range output {
		var s string
		if i == 0 {
			s = CharUnknown
		} else if i == len(g.Dataset.Chars)+1 {
			s = CharLinefeed
		} else {
			s = g.IndexToChar[i]
		}
		chars[i] = Char{S: s, Probability: o}
	}
	sort.Sort(ByProbabilityDesc(chars))
	return chars
}

type Char struct {
	S           string
	Probability float64
}

const (
	CharUnknown  = "Unknown"
	CharLinefeed = "Linefeed"
)

func (c Char) String() string {
	return fmt.Sprintf("{%s %.3g}", c.S, c.Probability)
}

type ByProbabilityDesc []Char

func (a ByProbabilityDesc) Len() int           { return len(a) }
func (a ByProbabilityDesc) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByProbabilityDesc) Less(i, j int) bool { return a[i].Probability >= a[j].Probability }
