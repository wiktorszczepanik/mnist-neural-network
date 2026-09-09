package network

type Layer struct {
	Length  int
	Neurons []Neuron
}

func PrepareLayer(layer *Layer, newLayerLength int, previousLayerLength int) {
	layer.Length = newLayerLength
	layer.Neurons = make([]Neuron, newLayerLength)
	for i := range newLayerLength {
		RandomNeuronInitValues(&layer.Neurons[i], previousLayerLength)
	}
}
