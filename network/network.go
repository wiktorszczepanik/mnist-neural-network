package network

import (
	"encoding/json"
	"errors"
	"log"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

var rng = rand.New(rand.NewSource(42))

type NeuralNetwork struct {
	LearningRate float64
	LayerSizes   []int
	Layers       []Layer
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
	slog.Info("Neural Network setup",
		"learning_rate", learningRate,
		"neurons", layerSizes,
		"layers", len(layerSizes),
	)

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

func (net *NeuralNetwork) Predict(pattern Pattern) (outputValue int, err error) {
	if err := validatePattern(net.LayerSizes, pattern); err != nil {
		return -1, err
	}
	outputLayer, _ := net.execute(pattern)
	var maxValue float64
	for index, value := range outputLayer {
		if value > maxValue {
			maxValue = value
			outputValue = index
		}
	}
	return outputValue, nil

}

func (net *NeuralNetwork) execute(pattern Pattern) (outputLayer []float64, err error) {
	// Set init layer with pattern values
	for i := range len(pattern.Features) {
		net.Layers[0].Neurons[i].Value = pattern.Features[i]
	}
	// Execute hidden & outputs
	var value float64
	for layerIndex := 1; layerIndex < len(net.LayerSizes); layerIndex++ {
		for neuronIndex := 0; neuronIndex < net.Layers[layerIndex].Length; neuronIndex++ {
			value = 0.0
			for link := 0; link < net.Layers[layerIndex-1].Length; link++ {
				value += net.Layers[layerIndex].Neurons[neuronIndex].Weights[link] * net.Layers[layerIndex-1].Neurons[link].Value
			}
			value += net.Layers[layerIndex].Neurons[neuronIndex].Bias
			//net.Layers[layerIndex].Neurons[neuronIndex].Value = Sigmoid(value)
			net.Layers[layerIndex].Neurons[neuronIndex].Value = ReLU(value)
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

func (net *NeuralNetwork) backPropagate(pattern Pattern) (deltaError float64) {
	outputLayer, _ := net.execute(pattern)
	var countedError float64 = 0.0
	// Output layer
	for neuronIndex := 0; neuronIndex < net.Layers[len(net.LayerSizes)-1].Length; neuronIndex++ {
		countedError = pattern.MultipleExpectation[neuronIndex] - outputLayer[neuronIndex]
		//net.Layers[len(net.LayerSizes)-1].Neurons[neuronIndex].Delta = countedError * SigmoidDerivative(outputLayer[neuronIndex])
		net.Layers[len(net.LayerSizes)-1].Neurons[neuronIndex].Delta = countedError * ReLUDerivative(outputLayer[neuronIndex])
	}
	// Left layers
	for layerIndex := len(net.LayerSizes) - 2; layerIndex >= 0; layerIndex-- {
		for neuronIndex := 0; neuronIndex < net.LayerSizes[layerIndex]; neuronIndex++ {
			countedError = 0.0
			for link := 0; link < net.Layers[layerIndex+1].Length; link++ {
				countedError += net.Layers[layerIndex+1].Neurons[link].Delta * net.Layers[layerIndex+1].Neurons[link].Weights[neuronIndex]
			}
			//net.Layers[layerIndex].Neurons[neuronIndex].Delta = countedError * SigmoidDerivative(net.Layers[layerIndex].Neurons[neuronIndex].Value)
			net.Layers[layerIndex].Neurons[neuronIndex].Delta = countedError * ReLUDerivative(net.Layers[layerIndex].Neurons[neuronIndex].Value)
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
	for i := range len(pattern.MultipleExpectation) {
		deltaError += math.Abs(outputLayer[i] - pattern.MultipleExpectation[i])
	}
	deltaError = deltaError / float64(len(pattern.MultipleExpectation))
	return deltaError
}

func (net *NeuralNetwork) Train(patterns []Pattern, epochs int) {
	slog.Info("Neural Network training", "pattern_elements", len(patterns), "epochs", epochs)
	for epoch := range epochs {
		shuffle(patterns)
		for _, pattern := range patterns {
			net.backPropagate(pattern)
		}
		log.Printf("epoch [%d/%d]", epoch+1, epochs)
	}
}

func shuffle(patterns []Pattern) {
	// Fisher Yates shuffle
	for i := len(patterns) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		patterns[i], patterns[j] = patterns[j], patterns[i]
	}

}

func (net *NeuralNetwork) Save(path string) error {
	data, err := json.Marshal(net)
	if err != nil {
		slog.Error("serialization process", "file", path, "error", err)
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		slog.Error("creating directory", "directory", dir, "error", err)
		return err
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		slog.Error("saving file", "file", path, "error", err)
		return err
	}
	slog.Info("saved model", "file", path)
	return nil
}

func Load(path string) (net *NeuralNetwork, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		slog.Error("read file", "file", path)
		return nil, err
	}
	err = json.Unmarshal(data, &net)
	if err != nil {
		slog.Error("deserialization process", "file", path)
		return nil, err
	}
	slog.Info("load model", "file", path)
	return net, nil
}

func (net *NeuralNetwork) Test(patterns []Pattern) (accuracy float64) {
	slog.Info("Neural Network testing", "pattern_elements", len(patterns))
	allPredictions := len(patterns)
	var correctPredictions int = 0
	for _, pattern := range patterns {
		correct := pattern.GetExpected()
		predict, _ := net.Predict(pattern)
		if predict == correct {
			correctPredictions++
		}
	}
	accuracy = float64(correctPredictions) / float64(allPredictions)
	slog.Info("accuracy", "value", accuracy*100)
	return accuracy
}

func (net *NeuralNetwork) TrainAndTest(patterns []Pattern, epochs int, testPatterns []Pattern) {
	slog.Info("Neural Network training", "pattern_elements", len(patterns), "epochs", epochs)
	for epoch := range epochs {
		shuffle(patterns)
		for _, pattern := range patterns {
			net.backPropagate(pattern)
		}
		log.Printf("epoch [%d/%d]", epoch+1, epochs)
		net.Test(testPatterns)
	}
}
