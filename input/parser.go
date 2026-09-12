package input

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
	"log/slog"
	"os"
)

type Image struct {
	Label  byte
	Pixels []byte
}

func GetImages(labelPath, pixelsPath string) (images []Image, err error) {
	labels, err := loadLabels(labelPath)
	if err != nil {
		return nil, err
	}
	pixels, step, err := loadPixels(pixelsPath)
	if err != nil {
		return nil, err
	}
	images = make([]Image, len(labels))
	for i, label := range labels {
		start := i * step
		end := start + step
		images[i] = Image{label, pixels[start:end]}
	}
	slog.Info("loaded", "labels", labelPath, "pixels", pixelsPath)
	return images, nil
}

func loadPixels(path string) (pixels []byte, step int, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, 0, err
	}
	defer gz.Close()
	var magic, count, rows, cols uint32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &count); err != nil {
		return nil, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
		return nil, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
		return nil, 0, err
	}
	if magic != 2051 {
		return nil, 0, errors.New("invalid image magic number")
	}
	step = int(rows * cols)
	pixels = make([]byte, step*int(count))
	if _, err := io.ReadFull(gz, pixels); err != nil {
		return nil, 0, err
	}
	return pixels, step, nil
}

func loadLabels(path string) (labels []byte, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	var magic, count uint32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(gz, binary.BigEndian, &count); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, errors.New("invalid image magic number")
	}
	labels = make([]byte, int(count))
	if _, err := io.ReadFull(gz, labels); err != nil {
		return nil, err
	}
	return labels, nil

}
