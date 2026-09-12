package network

import "artificial-neural-network/input"

// Pattern MNIST dataset
type Pattern struct {
	Features            [784]float64
	MultipleExpectation [10]float64
}

func (pattern Pattern) GetExpected() (outputValue int) {
	var maxValue float64
	for index, value := range pattern.MultipleExpectation {
		if value > maxValue {
			maxValue = value
			outputValue = index
		}
	}
	return outputValue
}

func ConvertFromImages(images []input.Image) (patterns []Pattern) {
	patterns = make([]Pattern, len(images))
	for i := range images {
		// Images Pixels to Pattern Features
		for j := range images[i].Pixels {
			patterns[i].Features[j] = float64(images[i].Pixels[j]) / 255.0
		}
		// Images Label to Pattern Expectations
		for j := range 10 {
			if int(images[i].Label) == j {
				patterns[i].MultipleExpectation[j] = 1
			} else {
				patterns[i].MultipleExpectation[j] = 0
			}
		}
	}
	return patterns
}

func ConvertFromImage(image input.Image) (pattern Pattern) {
	// Images Pixels to Pattern Features
	for i := range image.Pixels {
		pattern.Features[i] = float64(image.Pixels[i]) / 255.0
	}
	return pattern
}
