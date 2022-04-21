.PHONY: all

all: run

run:
	python3 src/main.py

test:
	python src/raspberry_inference.py --model models/dscnn88.tflite --audio record

