import argparse
import threading
import time
import os
from typing import Tuple

import numpy as np
from PIL import Image


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size < 3 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be an odd integer >= 3.")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel.astype(np.float32)


def laplacian_kernel() -> np.ndarray:
    return np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=np.float32,
    )


def _make_worker(
    padded: np.ndarray,
    kernel: np.ndarray,
    output: np.ndarray,
):
    kernel_height, kernel_width = kernel.shape
    out_height, out_width = output.shape

    def worker(row_start: int, row_end: int) -> None:
        for y in range(row_start, row_end):
            for x in range(out_width):
                region = padded[y:y + kernel_height, x:x + kernel_width]
                output[y, x] = np.sum(region * kernel)

    return worker


def multi_threaded_convolution(
    image: np.ndarray,
    kernel: np.ndarray,
    num_threads: int,
    padding_mode: str = "edge",
) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=padding_mode)
    output = np.zeros_like(image, dtype=np.float32)
    worker = _make_worker(padded, kernel, output)

    total_rows = output.shape[0]
    thread_count = max(1, min(num_threads, total_rows))
    chunk = max(1, (total_rows + thread_count - 1) // thread_count)

    threads = []
    for i in range(thread_count):
        start = i * chunk
        end = min(total_rows, (i + 1) * chunk)
        if start >= end:
            break
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return output


def zero_crossing_edges(laplacian: np.ndarray, threshold: float) -> np.ndarray:
    if threshold < 0:
        raise ValueError("Zero-crossing threshold must be non-negative.")

    padded = np.pad(laplacian, ((1, 1), (1, 1)), mode="edge")
    edges = np.zeros_like(laplacian, dtype=np.uint8)
    height, width = laplacian.shape

    for y in range(height):
        for x in range(width):
            neighborhood = padded[y:y + 3, x:x + 3]
            min_val = np.min(neighborhood)
            max_val = np.max(neighborhood)
            if min_val < 0 < max_val and (max_val - min_val) >= threshold:
                edges[y, x] = 255

    return edges


def apply_log(
    image: np.ndarray,
    sigma: float,
    kernel_size: int,
    threshold: float,
    num_threads: int,
) -> np.ndarray:
    gaussian = gaussian_kernel(kernel_size, sigma)
    smoothed = multi_threaded_convolution(image, gaussian, num_threads)
    laplacian = multi_threaded_convolution(smoothed, laplacian_kernel(), num_threads)
    return zero_crossing_edges(laplacian, threshold)


def save_grayscale_image(image: np.ndarray, output_path: str) -> None:
    Image.fromarray(image, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Laplacian of Gaussian edge detection.",
    )
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_dir", help="Path to save the edge image.")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for convolution.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma for smoothing.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Odd kernel size for Gaussian smoothing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Minimum Laplacian swing to keep a zero crossing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_grayscale_image(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    time_start = time.time()
    edges = apply_log(
        image=image,
        sigma=args.sigma,
        kernel_size=args.kernel_size,
        threshold=args.threshold,
        num_threads=max(1, args.threads),
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    output_image = os.path.join(args.output_dir, os.path.basename(args.input_image))
    save_grayscale_image(edges, output_image)


if __name__ == "__main__":
    main()

