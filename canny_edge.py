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


def sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32,
    )
    kernel_y = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
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


def non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude, dtype=np.float32)

    angle = (direction + 180.0) % 180.0

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current = magnitude[y, x]
            theta = angle[y, x]

            if (0.0 <= theta < 22.5) or (157.5 <= theta <= 180.0):
                neighbor1 = magnitude[y, x - 1]
                neighbor2 = magnitude[y, x + 1]
            elif 22.5 <= theta < 67.5:
                neighbor1 = magnitude[y - 1, x + 1]
                neighbor2 = magnitude[y + 1, x - 1]
            elif 67.5 <= theta < 112.5:
                neighbor1 = magnitude[y - 1, x]
                neighbor2 = magnitude[y + 1, x]
            else:
                neighbor1 = magnitude[y - 1, x - 1]
                neighbor2 = magnitude[y + 1, x + 1]

            if current >= neighbor1 and current >= neighbor2:
                suppressed[y, x] = current

    return suppressed


def double_threshold(
    magnitude: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> Tuple[np.ndarray, int, int]:
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be non-negative.")
    if high_threshold < low_threshold:
        raise ValueError("High threshold must be >= low threshold.")

    strong = 255
    weak = 75

    edges = np.zeros_like(magnitude, dtype=np.uint8)
    strong_mask = magnitude >= high_threshold
    weak_mask = (magnitude >= low_threshold) & ~strong_mask

    edges[strong_mask] = strong
    edges[weak_mask] = weak
    return edges, weak, strong


def hysteresis(edges: np.ndarray, weak: int, strong: int) -> np.ndarray:
    height, width = edges.shape
    result = edges.copy()

    strong_positions = list(zip(*np.where(result == strong)))
    stack = strong_positions[:]

    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if result[ny, nx] == weak:
                    result[ny, nx] = strong
                    stack.append((ny, nx))

    result[result != strong] = 0
    return result


def save_grayscale_image(image: np.ndarray, output_path: str) -> None:
    Image.fromarray(image, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Canny edge detection.",
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
        "--low-threshold",
        type=float,
        default=50.0,
        help="Low threshold for hysteresis (0-255 scale).",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=100.0,
        help="High threshold for hysteresis (0-255 scale).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Canny edge detection pipeline."""
    args = parse_args()
    image = load_grayscale_image(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)

    time_start = time.time()
    gaussian = gaussian_kernel(args.kernel_size, args.sigma)
    smoothed = multi_threaded_convolution(image, gaussian, max(1, args.threads))

    kernel_x, kernel_y = sobel_kernels()
    grad_x = multi_threaded_convolution(smoothed, kernel_x, max(1, args.threads))
    grad_y = multi_threaded_convolution(smoothed, kernel_y, max(1, args.threads))

    magnitude = np.hypot(grad_x, grad_y)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    direction = np.degrees(np.arctan2(grad_y, grad_x))

    suppressed = non_maximum_suppression(magnitude, direction)
    thresholded, weak, strong = double_threshold(
        suppressed,
        args.low_threshold,
        args.high_threshold,
    )
    edges = hysteresis(thresholded, weak, strong)

    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_image))
    save_grayscale_image(edges.astype(np.uint8), output_path)


if __name__ == "__main__":
    main()
