// Package nn implements a simple neural network with
// back-propagation. This network has two input nodes,
// a hidden layer, and a single output node.
package nnet

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const DefaultLearningRate = 0.2

// A NeuralNetwork implements a neural network with a single hidden
// layer.
type NeuralNetwork struct {
	inputs       []float64
	inputWeights [][]float64

	// TODO(kyle): make hidden a slice of slices to support multiple
	// layers.
	hidden           []float64
	hiddenThresholds []float64
	hiddenWeights    [][]float64

	outputs          []float64
	outputThresholds []float64

	learningRate float64
}

func initMatrix(x, y int) [][]float64 {
	m := make([][]float64, x)
	for i := 0; i < x; i++ {
		m[i] = make([]float64, y)
	}

	return m
}

// New produces a neural network with the specified number of input,
// hidden, and output neurons.
func New(nInputs, nHidden, nOutputs int) *NeuralNetwork {
	net := &NeuralNetwork{
		inputs:           make([]float64, nInputs),
		hidden:           make([]float64, nHidden),
		hiddenThresholds: make([]float64, nHidden),
		outputs:          make([]float64, nOutputs),
		outputThresholds: make([]float64, nOutputs),
		inputWeights:     initMatrix(nInputs, nHidden),
		hiddenWeights:    initMatrix(nHidden, nOutputs),
		learningRate:     DefaultLearningRate,
	}
	net.connect()
	return net
}

// LearningRate returns the current learning rate. If 0 < lr < 1, the
// learning rate for the network will be set to lr.
func (nn *NeuralNetwork) LearningRate(lr float64) float64 {
	current := nn.learningRate
	if lr > 0 && lr < 1 {
		nn.learningRate = lr
	}
	return current
}

func (nn *NeuralNetwork) connect() {
	for i := 0; i < len(nn.inputs); i++ {
		for h := 0; h < len(nn.hidden); h++ {
			nn.inputWeights[i][h] = rand()
		}
	}

	for h := 0; h < len(nn.hidden); h++ {
		for o := 0; o < len(nn.outputs); o++ {
			nn.hiddenWeights[h][o] = rand()
			nn.outputThresholds[o] = rand() / rand()

		}
		nn.hiddenThresholds[h] = rand() / rand()
	}
}

// Input provides a set of inputs to the neural network.
func (nn *NeuralNetwork) Input(inputs ...float64) error {
	if len(inputs) != len(nn.inputs) {
		return errors.New("invalid inputs to neural network")
	}

	for i := 0; i < len(inputs); i++ {
		nn.inputs[i] = inputs[i]
	}

	return nil
}

// Activate causes the neural network to process its inputs.
func (nn *NeuralNetwork) Activate() {
	var weighted float64

	for h := 0; h < len(nn.hidden); h++ {
		weighted = 0.0
		for i := 0; i < len(nn.inputs); i++ {
			weighted += nn.inputWeights[i][h] * nn.inputs[i]
		}

		weighted -= nn.hiddenThresholds[h]
		nn.hidden[h] = 1.0 / (1.0 + math.Pow(math.E, -weighted))
	}

	for o := 0; o < len(nn.outputs); o++ {
		weighted = 0.0
		for h := 0; h < len(nn.hidden); h++ {
			weighted += nn.hiddenWeights[h][o] * nn.hidden[h]
		}

		weighted -= nn.outputThresholds[o]
		nn.outputs[o] = 1.0 / (1.0 + math.Pow(math.E, -weighted))
	}
}

// Train backpropagates and updates the network's weights and
// thresholds based on the expected values provided.
func (nn *NeuralNetwork) Train(expected ...float64) (float64, error) {
	if len(expected) != len(nn.outputs) {
		return 0, fmt.Errorf("invalid expected output: wanted %d expected values but have %d",
			len(nn.outputs), len(expected))
	}

	var sosError, absError, oGradient, hGradient, delta, tDelta float64
	for o := 0; o < len(nn.outputs); o++ {
		absError = expected[o] - nn.outputs[o]
		sosError += math.Pow(absError, 2)
		oGradient = nn.outputs[o] * (1.0 - nn.outputs[o]) * absError

		for h := 0; h < len(nn.hidden); h++ {
			delta = nn.learningRate * nn.hidden[h] * oGradient
			nn.hiddenWeights[h][o] += delta
			hGradient = nn.hidden[h] * (1 - nn.hidden[h]) * oGradient * nn.hiddenWeights[h][o]

			for i := 0; i < len(nn.inputs); i++ {
				delta = nn.learningRate * nn.inputs[i] * hGradient
				nn.inputWeights[i][h] += delta
			}

			tDelta = nn.learningRate * hGradient * -1
			nn.hiddenThresholds[h] += tDelta
		}

		delta = nn.learningRate * oGradient * -1
		nn.outputThresholds[o] += delta
	}

	return sosError, nil
}

// Results collects the values of the network's output neurons.
func (nn *NeuralNetwork) Results() []float64 {
	var results = make([]float64, len(nn.outputs))
	for o := range nn.outputs {
		results[o] = nn.outputs[o]
	}
	return results
}

// StateLine returns the state of the input and output nodes as a
// string.
func (nn *NeuralNetwork) StateLine() string {
	var pieces []string

	for i := 0; i < len(nn.inputs); i++ {
		pieces = append(pieces, fmt.Sprintf(" IN%d: %2.4f", i+1, nn.inputs[i]))
	}

	for o := 0; o < len(nn.outputs); o++ {
		pieces = append(pieces, fmt.Sprintf("ON%d: %2.4f", o+1, nn.outputs[o]))
	}

	return strings.Join(pieces, " | ")
}
