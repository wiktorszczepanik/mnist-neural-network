package network

//type NeuralNetworkOperations interface {
//}

type NeuralNetwork struct {
	LearningRate float64
	Layers       []Layer
}

func NewNeuralNetwork(layerSizes []int, learningRate float64) (neuralNetwork NeuralNetwork) {
	neuralNetwork.LearningRate = learningRate
	neuralNetwork.Layers = make([]Layer, len(layerSizes))
	for index, size := range layerSizes {
		if index != 0 {
			PrepareLayer(&neuralNetwork.Layers[index], size, index-1)
		} else {
			PrepareLayer(&neuralNetwork.Layers[index], size, index)
		}
	}
	return neuralNetwork
}
