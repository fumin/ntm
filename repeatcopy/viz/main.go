package main

import (
	"log"

	"github.com/fumin/ntm/repeatcopy"
)

func main() {
	x, y := repeatcopy.G["bt"](2, 3)
	log.Printf("x: %+v", x)
	log.Printf("y: %+v", y)
}
