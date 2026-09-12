package input

import (
	"errors"
	"image"
	"image/color"
	_ "image/png"
	"os"
)

func GetImage(inputImage string) (outputImage Image, err error) {
	file, err := os.Open(inputImage)
	if err != nil {
		return outputImage, err
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return outputImage, err
	}
	bounds := img.Bounds()
	if bounds.Max.X != 28 || bounds.Max.Y != 28 {
		return outputImage, errors.New("invalid image size")
	}
	gray := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			gray.Set(x, y, color.GrayModel.Convert(img.At(x, y)))
		}
	}
	pixels := make([]byte, 0, bounds.Dx()*bounds.Dy())
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := gray.GrayAt(x, y).Y
			pixels = append(pixels, pixel)
		}
	}
	outputImage.Label = 0
	outputImage.Pixels = pixels
	return outputImage, nil
}
