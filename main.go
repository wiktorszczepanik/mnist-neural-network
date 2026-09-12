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

const ManualPngImagePath = "data/image.png"

const ModelPath = "output/model.json"

func main() {
	TrainAndSaveModel()
	LoadModelAndTest()
	//ManualPrediction()
}

// TrainAndSaveModel accuracy value=98.08
func TrainAndSaveModel() {
	images, _ := input.GetImages(TrainLabelsPath, TrainImagesPath)
	patterns := network.ConvertFromImages(images)
	nnParameters := []int{784, 128, 64, 10}
	learningRate := 0.005
	epochs := 20
	neuralNetwork := network.NewNeuralNetwork(nnParameters, learningRate)
	neuralNetwork.Train(patterns, epochs)
	if err := neuralNetwork.Save(ModelPath); err != nil {
		return
	}
}

func LoadModelAndTest() {
	images, _ := input.GetImages(TestLabelsPath, TestImagesPath)
	patterns := network.ConvertFromImages(images)
	neuralNetwork, err := network.Load(ModelPath)
	if err != nil {
		return
	}
	neuralNetwork.Test(patterns)
}

func ManualPrediction() {
	image, err := input.GetImage(ManualPngImagePath)
	if err != nil {
		return
	}
	pattern := network.ConvertFromImage(image)
	neuralNetwork, err := network.Load(ModelPath)
	if err != nil {
		return
	}
	value, err := neuralNetwork.Predict(pattern)
	if err != nil {
		return
	}
	log.Printf("Prediction: %d", value)
}
