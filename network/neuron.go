package network

import (
	"math"
	"math/rand"
)

const ScalingFactor = 0.1

type Neuron struct {
	Weights []float64
	Bias    float64
	Value   float64
	Delta   float64
}

func RandomNeuronInitValues(neuron *Neuron, weightsNumber int) {
	neuron.Weights = make([]float64, weightsNumber)
	heInit(neuron, weightsNumber)
	neuron.Value = 0
	neuron.Delta = 0
}

func heInit(neuron *Neuron, weightsNumber int) {
	stddev := math.Sqrt(2.0 / float64(weightsNumber))
	for i := range weightsNumber {
		neuron.Weights[i] = rand.NormFloat64() * stddev
	}
	neuron.Bias = 0
}

func gaussianInit(neuron *Neuron, weightsNumber int) {
	neuron.Weights = make([]float64, weightsNumber)
	for i := range weightsNumber {
		neuron.Weights[i] = rand.NormFloat64() * ScalingFactor
	}
	neuron.Bias = rand.NormFloat64() * ScalingFactor
}
