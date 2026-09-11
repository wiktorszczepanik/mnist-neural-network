package main

import (
	"artificial-neural-network/input"
	"artificial-neural-network/network"
	"log"
)

const TrainLabelsPath = "data/train-labels-idx1-ubyte.gz"
const TrainImagesPath = "data/train-images-idx3-ubyte.gz"

const TestLabelsPath = "data/t10k-labels-idx1-ubyte.gz"
const TestImagesPath = "data/t10k-images-idx3-ubyte.gz"

const ModelPath = "output/model.json"

func main() {
	TrainAndSaveModel()
	LoadModelAndTest()
}

// Accuracy: 97.50%
func TrainAndSaveModel() {
	images, _ := input.GetImages(TrainLabelsPath, TrainImagesPath)
	log.Printf("Loaded train images: \n\t%s\n\t%s", TrainLabelsPath, TrainImagesPath)
	patterns := network.ConvertFromImages(images)
	nnParameters := []int{784, 128, 64, 10} // {784, 128, 64, 10}
	learningRate := 0.01                    // 0.01
	epochs := 40                            // 20 // 25 // 30 // 40
	log.Printf(
		"Neural Network info:\n"+"\tLearning rate: %.4f\n"+"\tEpochs: %d\n"+"\tNeurons: %v\n"+"\tLayers: %d",
		learningRate, epochs, nnParameters, len(nnParameters),
	)
	neuralNetwork := network.NewNeuralNetwork(nnParameters, learningRate)
	neuralNetwork.Train(patterns, epochs)
	if err := neuralNetwork.Save(ModelPath); err != nil {
		return
	}
	log.Printf("Saved model: %s", ModelPath)
}

func LoadModelAndTest() {
	images, _ := input.GetImages(TestLabelsPath, TestImagesPath)
	log.Printf("Loaded test images: \n\t%s\n\t%s", TestLabelsPath, TestImagesPath)
	patterns := network.ConvertFromImages(images)
	neuralNetwork, err := network.Load(ModelPath)
	if err != nil {
		return
	}
	log.Printf("Loaded model: %s", ModelPath)
	accuracy := neuralNetwork.Test(patterns)
	log.Printf("Accuracy: %.2f%%", accuracy*100)
}
