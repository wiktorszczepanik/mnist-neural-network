package network

import (
	"errors"
	"math"
)

type NeuralNetwork struct {
	LearningRate         float64
	LayerSizes           []int
	Layers               []Layer
	ActivationFunction   ActivationFunction
	ActivationDerivative ActivationDerivative
}

func NewNeuralNetwork(layerSizes []int, learningRate float64) (neuralNetwork NeuralNetwork) {
	neuralNetwork.LayerSizes = layerSizes
	neuralNetwork.LearningRate = learningRate
	neuralNetwork.Layers = make([]Layer, len(layerSizes))
	for index, size := range layerSizes {
		if index != 0 {
			PrepareLayer(&neuralNetwork.Layers[index], size, layerSizes[index-1])
		} else {
			PrepareLayer(&neuralNetwork.Layers[index], size, 0)
		}
	}
	neuralNetwork.ActivationFunction = Sigmoid
	neuralNetwork.ActivationDerivative = SigmoidDerivative
	return neuralNetwork
}

func validatePattern(layerSizes []int, pattern Pattern) error {
	if len(pattern.Features) != layerSizes[0] {
		return errors.New("incompatible features with first layer")
	}
	if len(pattern.MultipleExpectation) != layerSizes[len(layerSizes)-1] {
		return errors.New("incompatible expected values with last layer")
	}
	return nil
}

func (net *NeuralNetwork) Predict(pattern Pattern) (outputLayer []float64, err error) {
	if err := validatePattern(net.LayerSizes, pattern); err != nil {
		return nil, err
	}
	outputLayer, _ = net.Execute(pattern)
	return outputLayer, nil

}

func (net *NeuralNetwork) Execute(pattern Pattern) (outputLayer []float64, err error) {
	// Set init layer with pattern values
	for i := 0; i < len(pattern.Features); i++ {
		net.Layers[0].Neurons[i].Value = pattern.Features[i]
	}
	// Execute hidden & outputs
	var value float64
	for layerIndex := 1; layerIndex < len(net.LayerSizes); layerIndex++ {
		for neuronIndex := 0; neuronIndex < net.Layers[layerIndex].Length; neuronIndex++ {
			value = 0.0
			for link := 0; link < net.Layers[layerIndex-1].Length; link++ {
				value += net.Layers[layerIndex].Neurons[neuronIndex].Weights[link] * net.Layers[layerIndex-1].Neurons[neuronIndex].Value
			}
			value += net.Layers[layerIndex].Neurons[neuronIndex].Bias
			net.Layers[layerIndex].Neurons[neuronIndex].Value = net.ActivationFunction(value)
		}
	}
	// Get output values
	outputLayer = make([]float64, len(pattern.MultipleExpectation))
	lastLayerIndex := len(net.Layers) - 1
	for i := range outputLayer {
		outputLayer[i] = net.Layers[lastLayerIndex].Neurons[i].Value
	}
	return outputLayer, nil
}

func (net *NeuralNetwork) BackPropagate(pattern Pattern) (deltaError float64) {
	outputLayer, _ := net.Execute(pattern)
	var countedError float64 = 0.0
	// Output layer
	for neuronIndex := 0; neuronIndex < net.Layers[len(net.LayerSizes)-1].Length; neuronIndex++ {
		countedError = pattern.MultipleExpectation[neuronIndex] - outputLayer[neuronIndex]
		net.Layers[len(net.LayerSizes)-1].Neurons[neuronIndex].Delta = countedError * net.ActivationDerivative(outputLayer[neuronIndex])
	}
	// Left layers
	for layerIndex := len(net.LayerSizes) - 2; layerIndex >= 0; layerIndex-- {
		for neuronIndex := 0; neuronIndex < net.LayerSizes[layerIndex]; neuronIndex++ {
			countedError = 0.0
			for link := 0; link < net.Layers[layerIndex+1].Length; link++ {
				countedError += net.Layers[layerIndex+1].Neurons[link].Delta * net.Layers[layerIndex+1].Neurons[link].Weights[neuronIndex]
			}
			net.Layers[layerIndex].Neurons[neuronIndex].Delta = countedError * net.ActivationDerivative(net.Layers[layerIndex].Neurons[neuronIndex].Value)
		}
		for neuronIndex := 0; neuronIndex < net.Layers[layerIndex+1].Length; neuronIndex++ {
			for link := 0; link < net.Layers[layerIndex].Length; link++ {
				net.Layers[layerIndex+1].Neurons[neuronIndex].Weights[link] +=
					net.LearningRate * net.Layers[layerIndex+1].Neurons[neuronIndex].Delta * net.Layers[layerIndex].Neurons[link].Value
			}
			net.Layers[layerIndex+1].Neurons[neuronIndex].Bias += net.LearningRate * net.Layers[layerIndex+1].Neurons[neuronIndex].Delta
		}
	}
	// Global errors as sum of abs difference
	for i := 0; i < len(pattern.MultipleExpectation); i++ {
		deltaError += math.Abs(outputLayer[i] - pattern.MultipleExpectation[i])
	}
	deltaError = deltaError / float64(len(pattern.MultipleExpectation))
	return deltaError
}

func (net *NeuralNetwork) Train(patterns []Pattern, epochs int) {
	for range epochs {
		for _, pattern := range patterns {
			net.BackPropagate(pattern)
		}
	}
}
