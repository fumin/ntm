package ntm

import (
	"math"
	"testing"
)

func TestRMSProp(t *testing.T) {
	xSize := 1
	ySize := 1
	h1Size := 1
	numHeads := 1
	n := 1
	m := 1
	c := NewEmptyController1(xSize, ySize, h1Size, numHeads, n, m)
	rms := NewRMSProp(c)

	c.WeightsVal()[0] = 1.1
	c.WeightsGrad()[0] = 2.7
	rms.N[0] = 10.3
	rms.G[0] = 1.8
	rms.D[0] = 3.7

	c.WeightsVal()[1] = 1.2
	c.WeightsGrad()[1] = 1.9
	rms.N[1] = 14.3
	rms.G[1] = 2.1
	rms.D[1] = 1.7

	c.WeightsVal()[len(c.WeightsVal())-1] = 0.9
	c.WeightsGrad()[len(c.WeightsVal())-1] = 1.3
	rms.N[len(c.WeightsVal())-1] = 12.3
	rms.G[len(c.WeightsVal())-1] = 0.8
	rms.D[len(c.WeightsVal())-1] = 8.1

	rms.update(0.95, 0.9, 0.0001, 0.0001)

	checkRMS(t, c, rms, 0, 10.1495, 1.845, 3.329896, 4.429896)
	checkRMS(t, c, rms, 1, 13.7655, 2.09, 1.529938, 2.729938)
	checkRMS(t, c, rms, len(c.WeightsVal())-1, 11.7695, 0.825, 7.289961, 8.189961)
}

func checkRMS(t *testing.T, c Controller, rms *RMSProp, i int, n, g, d, w float64) {
	tol := 1e-6
	if math.Abs(rms.N[i]-n) > tol {
		t.Errorf("rms.N[%d](%g) != %g", i, rms.N[i], n)
	}
	if math.Abs(rms.G[i]-g) > tol {
		t.Errorf("rms.G[%d](%g) != %g", i, rms.G[i], g)
	}
	if math.Abs(rms.D[i]-d) > tol {
		t.Errorf("rms.D[%d](%g) != %g", i, rms.D[i], d)
	}
	if math.Abs(c.WeightsVal()[i]-w) > tol {
		t.Errorf("w[%d](%g) != %g", i, c.WeightsVal()[i], w)
	}
}
