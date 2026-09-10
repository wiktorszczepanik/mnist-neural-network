package main

import (
	"artificial-neural-network/input"
	"artificial-neural-network/network"
	"fmt"
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

func TrainAndSaveModel() {
	images, _ := input.GetImages(TrainLabelsPath, TrainImagesPath)
	patterns := network.ConvertFromImages(images)
	nnSize := []int{784, 128, 64, 10}
	neuralNetwork := network.NewNeuralNetwork(nnSize, 0.01)
	neuralNetwork.Train(patterns, 15)
	err := neuralNetwork.Save(ModelPath)
	if err != nil {
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
	accuracy := neuralNetwork.Test(patterns)
	fmt.Println("Accuracy: ", accuracy)

}

func LoadModelAndPredict() {
	//neuralNetwork, err := network.Load(ModelPath)
	//if err != nil {
	//	return
	//}
	//images, _ := input.GetImages(TrainLabelsPath, TrainImagesPath)
	//patterns := network.ConvertFromImages(images)
	//output, err := neuralNetwork.Predict()
}
