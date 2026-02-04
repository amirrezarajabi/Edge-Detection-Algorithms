import argparse
import threading
import time
from typing import Tuple

import numpy as np
from PIL import Image
import os


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32)


def roberts_cross_kernels() -> Tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array(
        [[1, 0],
         [0, -1]],
        dtype=np.float32,
    )
    kernel_y = np.array(
        [[0, 1],
         [-1, 0]],
        dtype=np.float32,
    )
    return kernel_x, kernel_y


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


def save_grayscale_image(image: np.ndarray, output_path: str) -> None:
    Image.fromarray(image, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Roberts Cross edge detection using threaded convolution.",
    )
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_dir", help="Path to save the edge image.")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for convolution.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Roberts Cross edge detection pipeline."""
    args = parse_args()
    image = load_grayscale_image(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)

    time_start = time.time()
    kernel_x, kernel_y = roberts_cross_kernels()
    grad_x = multi_threaded_convolution(image, kernel_x, max(1, args.threads))
    grad_y = multi_threaded_convolution(image, kernel_y, max(1, args.threads))

    magnitude = np.hypot(grad_x, grad_y)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    edges = magnitude.astype(np.uint8)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_image))
    save_grayscale_image(edges, output_path)

def run_roberts_edge(image_path: str) -> np.ndarray:
    threads = 8
    image = load_grayscale_image(image_path)
    kernel_x, kernel_y = roberts_cross_kernels()
    grad_x = multi_threaded_convolution(image, kernel_x, threads)
    grad_y = multi_threaded_convolution(image, kernel_y, threads)
    magnitude = np.hypot(grad_x, grad_y)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    return magnitude.astype(np.float32) / 255.0

if __name__ == "__main__":
    main()

