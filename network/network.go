package network

import "errors"

type NeuralNetworkOperations interface {
	Execute(pattern Pattern) (outputLayer []float64, err error)
}

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

func (net *NeuralNetwork) Execute(pattern Pattern) (outputLayer []float64, err error) {
	if err := validatePattern(net.LayerSizes, pattern); err != nil {
		return nil, err
	}
	// Set init layer with pattern values
	for i := 0; i < len(pattern.Features); i++ {
		net.Layers[0].Neurons[i].Value = pattern.Features[i]
	}
	// Execute hidden & outputs
	var value float64
	for layerIndex := 1; layerIndex < len(net.LayerSizes); layerIndex++ {
		for neuronIndex := 0; neuronIndex < net.Layers[layerIndex].Length; neuronIndex++ {
			value = 0.0
			for connection := 0; connection < net.Layers[layerIndex-1].Length; connection++ {
				value += net.Layers[layerIndex].Neurons[neuronIndex].Weights[connection] * net.Layers[layerIndex-1].Neurons[neuronIndex].Value
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
	return outputLayer, err
}
