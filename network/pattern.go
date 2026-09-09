package network

type Pattern struct {
	Features            []float64
	MultipleExpectation []float64
}

func (pattern Pattern) scaleFeatures() {
	//for i := range pattern.Features {
	//	pattern.Features[i] = Sigmoid(pattern.Features[i])
	//}
}
