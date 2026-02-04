# Edge Detection Algorithms

This project runs classic edge detection algorithms on sample images and writes
the results into per-algorithm output folders.

## Setup

1. Create a virtual environment (optional).
2. Install dependencies:

```
pip install -r requirements.txt
```

## Run

Each script takes an input image, an output directory, and optional parameters.
Outputs are written into the target directory with the same filename as the
input image.

### Prewitt

```
python prewitt_edge.py images/face.jpg results/prewitt --threads 8
```

### Sobel

```
python sobel_edge.py images/face.jpg results/sobel --threads 8
```

### Roberts

```
python roberts_edge.py images/face.jpg results/roberts --threads 8
```

### LoG (Laplacian of Gaussian)

```
python log_edge.py images/face.jpg results/log --threads 8 --sigma 1.4 --kernel-size 5 --threshold 10
```

### Canny

```
python canny_edge.py images/face.jpg results/canny --threads 8 --sigma 1.4 --kernel-size 5 --low-threshold 50 --high-threshold 100
```

## Outputs

- Results are stored under `results/` by algorithm.
- Filenames mirror the input images.
