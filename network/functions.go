package network

import "math"

func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Pow(math.E, -value))
}

func SigmoidDerivative(value float64) float64 {
	return value * (1 - value)
}
