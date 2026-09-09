package network

import "math"

type ActivationFunction func(float64) float64
type ActivationDerivative func(float64) float64

func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Pow(math.E, -value))
}

func SigmoidDerivative(value float64) float64 {
	sigmoid := Sigmoid(value)
	return sigmoid * (1 - sigmoid)
}
