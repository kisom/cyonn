// This is an implementation of the XOR neural network from "Code Your
// Own Neural Network" by Steven C. Shaffer.
package main

import (
	"flag"
	"fmt"
	"os"

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

func XORTrainUntilSuccess(updateAt int, net *nnet.NeuralNetwork) {
	var iterations int

	for {
		display := (iterations % updateAt) == 0
		var printEvery int

		if display {
			printEvery = 1
		}
		_, err := XORTrainingSet(1, printEvery, net)
		success := XORTest(net)
		fmt.Printf("%10d> SUCC: %3.2f\t ERR: %8.5f\n", iterations, success, err)
		if success > 0.9 && err < 0.01 {
			break
		}

		iterations++
	}

	fmt.Printf("Network fully trained after %d iterations\n", iterations)
}

func main() {
	var iterations, printEvery, nets int
	var learnRate float64
	flag.IntVar(&iterations, "i", 131072, "number of training iterations")
	flag.Float64Var(&learnRate, "l", nnet.DefaultLearningRate, "learning rate")
	flag.IntVar(&nets, "n", 1, "number of neural networks to train")
	flag.IntVar(&printEvery, "p", 1024, "display step")
	trainToSuccess := flag.Bool("t", false, "train the network until it is successful")
	flag.Parse()

	if *trainToSuccess {
		for n := 0; n < nets; n++ {
			net := nnet.New(2, 2, 1)
			net.LearningRate(learnRate)
			fmt.Println("2-2-1 neural network initialised with learning rate", net.LearningRate(0))
			XORTrainUntilSuccess(printEvery, net)
			return
		}
	}

	var successRate float64
	for n := 0; n < nets; n++ {
		net := nnet.New(2, 2, 1)
		net.LearningRate(learnRate)
		fmt.Println("2-2-1 neural network initialised with learning rate", net.LearningRate(0))
		XORTrainingSet(iterations, printEvery, net)
		successRate += XORTest(net)
	}

	fmt.Printf("Average success rate for 2-2-1 network with a = %0.3f over %d iterations: %3.5f\n",
		learnRate, iterations, successRate/float64(nets))
}
