package nnet

import (
	_rand "crypto/rand"
	"encoding/binary"
	"fmt"
	mrand "math/rand"
	"os"
)

var seed int64

func init() {
	var seedBytes = make([]byte, 8)
	_, err := _rand.Read(seedBytes)
	if err != nil {
		fmt.Println("Failed to initialise random generator.")
		os.Exit(1)
	}

	seed = int64(binary.LittleEndian.Uint64(seedBytes))
	mrand.Seed(seed)
}

func rand() float64 {
	return mrand.Float64()
}
