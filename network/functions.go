package network

import "math"

func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Pow(math.E, -value))
}

func SigmoidDerivative(value float64) float64 {
	return value * (1 - value)
}

func ReLU(value float64) float64 {
	return math.Max(0, value)
}

func ReLUDerivative(value float64) float64 {
	if value > 0.0 {
		return 1.0
	} else {
		return 0.0
	}
}
