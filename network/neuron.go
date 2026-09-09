package network

import "math/rand"

const ScalingFactor = 0.1

type Neuron struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Value        float64
	Delta        float64
}

func RandomNeuronInitValues(neuron *Neuron, weightsNumber int) {
	neuron.Weights = make([]float64, weightsNumber)
	for i := range weightsNumber {
		neuron.Weights[i] = rand.NormFloat64() * ScalingFactor
	}
	neuron.Bias = rand.NormFloat64() * ScalingFactor
	neuron.LearningRate = rand.NormFloat64() * ScalingFactor
	neuron.Value = rand.NormFloat64() * ScalingFactor
	neuron.Delta = rand.NormFloat64() * ScalingFactor
}
