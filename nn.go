// This is an implementation of the XOR neural network from "Code Your
// Own Neural Network" by Steven C. Shaffer.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/kisom/cyonn/nnet"
)

func errorf(err error) {
	fmt.Fprintf(os.Stderr, "[!] %v\n", err)
}

func round(f float64) int {
	if f < 0.5 {
		return 0
	}
	return 1
}

func XORTrainingSet(iterations, printEvery int, net *nnet.NeuralNetwork) (int, float64) {
	var sosError float64
	var i int
	var display bool

	for i := 0; i < iterations; i++ {
		display = false
		sosError = 0.0
		if (printEvery != 0) && ((i % printEvery) == 0) {
			display = true
			fmt.Println("Neural network snapshot:")
		}

		net.Input(0, 0)
		net.Activate()
		sos, err := net.Train(0)
		if err != nil {
			errorf(err)
			return i, sosError
		}
		sosError += sos
		if display {
			fmt.Printf("%5d> %s | err: %2.3f\n", i, net.StateLine(), sos)
		}

		net.Input(0, 1)
		net.Activate()
		sos, err = net.Train(1)
		if err != nil {
			errorf(err)
			return i, sosError
		}
		sosError += sos
		if display {
			fmt.Printf("%5d> %s | err: %2.3f\n", i, net.StateLine(), sos)
		}

		net.Input(1, 0)
		net.Activate()
		sos, err = net.Train(1)
		if err != nil {
			errorf(err)
			return i, sosError
		}
		sosError += sos
		if display {
			fmt.Printf("%5d> %s | err: %2.3f\n", i, net.StateLine(), sos)
		}

		net.Input(1, 1)
		net.Activate()
		net.Train(0)
		sos, err = net.Train(0)
		if err != nil {
			errorf(err)
			return i, sosError
		}
		sosError += sos
		if display {
			fmt.Printf("%5d> %s | err: %2.3f\n", i, net.StateLine(), sos)
		}

		if display {
			fmt.Println("------------------------------------------------------------------------")
		}

		sosError /= 4.0
	}

	return i, sosError
}

func XORTest(net *nnet.NeuralNetwork) float64 {
	success := 0

	net.Input(0, 0)
	net.Activate()
	out := net.Results()
	if round(out[0]) == 0 {
		success++
	}

	net.Input(0, 1)
	net.Activate()
	out = net.Results()
	if round(out[0]) == 1 {
		success++
	}

	net.Input(1, 0)
	net.Activate()
	out = net.Results()
	if round(out[0]) == 1 {
		success++
	}

	net.Input(1, 1)
	net.Activate()
	out = net.Results()
	if round(out[0]) == 0 {
		success++
	}

	return float64(success) / 4.0
}

func improvement(cur, prev float64) float64 {
	return math.Abs((cur - prev) / cur)
}

func XORTrainUntilSuccess(updateAt int, net *nnet.NeuralNetwork) int {
	var iterations int
	var lastChangeAt int
	var lastErr float64

	for {
		display := (updateAt != 0) && (iterations%updateAt) == 0
		var printEvery int

		if display {
			printEvery = 1
		}
		_, err := XORTrainingSet(1, printEvery, net)
		success := XORTest(net)
		if success > 0.9 && err < 0.01 {
			break
		}

		if improvement(err, lastErr) > 0.001 {
			lastChangeAt = iterations
			lastErr = err
		} else {
			if (iterations - lastChangeAt) > 10000 {
				fmt.Println("*** Stagnation detected: no change since",
					lastChangeAt, "with current generation",
					iterations)
				fmt.Printf("%10d> SUCC: %3.2f\t ERR: %8.5f\n", iterations, success, err)
				return -1
			}
		}

		iterations++
	}

	fmt.Printf("Network fully trained after %d iterations\n", iterations)
	return iterations
}

func parseConfig(config string) *nnet.NeuralNetwork {
	confElts := strings.Split(config, "-")
	if len(confElts) != 3 {
		return nnet.New(2, 2, 1)
	}

	inputs, err := strconv.Atoi(confElts[0])
	if err != nil {
		return nnet.New(2, 2, 1)
	}

	hidden, err := strconv.Atoi(confElts[1])
	if err != nil {
		return nnet.New(2, 2, 1)
	}

	outputs, err := strconv.Atoi(confElts[2])
	if err != nil {
		return nnet.New(2, 2, 1)
	}

	fmt.Printf("%d-%d-%d initialised ", inputs, hidden, outputs)
	return nnet.New(inputs, hidden, outputs)
}

func main() {
	var iterations, printEvery, nets int
	var learnRate float64
	var config string

	flag.StringVar(&config, "c", "2-2-1", "network configuration")
	flag.IntVar(&iterations, "i", 131072, "number of training iterations")
	flag.Float64Var(&learnRate, "l", nnet.DefaultLearningRate, "learning rate")
	flag.IntVar(&nets, "n", 1, "number of neural networks to train")
	flag.IntVar(&printEvery, "p", 1024, "display step")
	trainToSuccess := flag.Bool("t", false, "train the network until it is successful")
	flag.Parse()

	if *trainToSuccess {
		iterations = 0
		success := 0
		for n := 0; n < nets; n++ {
			net := parseConfig(config)
			net.LearningRate(learnRate)
			fmt.Println("with learning rate", net.LearningRate(0))
			iter := XORTrainUntilSuccess(printEvery, net)
			if iter != -1 {
				iterations += iter
				success++
			}
		}
		fmt.Printf("%s: mean success rate %5.3f with mean training time %d generations\n",
			config, float64(success)/float64(nets), iterations/success)
		return
	}

	var successRate float64
	for n := 0; n < nets; n++ {
		net := parseConfig(config)
		net.LearningRate(learnRate)
		fmt.Println("with learning rate", net.LearningRate(0))
		XORTrainingSet(iterations, printEvery, net)
		successRate += XORTest(net)
	}

	fmt.Printf("Average success rate for %s network with a = %0.3f over %d iterations: %3.5f\n",
		config, learnRate, iterations, successRate/float64(nets))
}
