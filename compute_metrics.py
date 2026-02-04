import os
from typing import List, Tuple
import numpy as np
from scipy.io import loadmat
from canny_edge import run_canny_edge
from sobel_edge import run_sobel_edge
from roberts_edge import run_roberts_edge
from prewitt_edge import run_prewitt_edge
from log_edge import run_log_edge
import argparse
from tqdm import tqdm

# default parameters
# LOG
LOG_THRESHOLD = 10.0
# CANNY
CANNY_LOW_THRESHOLD = 50.0
CANNY_HIGH_THRESHOLD = 100.0

def load_ground_truth_edges(mat_path: str) -> np.ndarray:
    """Load and aggregate ground-truth boundaries from a BSDS500 .mat file."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Ground truth file not found: {mat_path}")

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    ground_truth = data.get("groundTruth")
    if ground_truth is None:
        raise KeyError(f"Missing 'groundTruth' key in: {mat_path}")

    if isinstance(ground_truth, np.ndarray):
        entries = ground_truth.ravel().tolist()
    else:
        entries = [ground_truth]

    boundaries: List[np.ndarray] = []
    for entry in entries:
        if hasattr(entry, "Boundaries"):
            boundaries.append(np.asarray(entry.Boundaries, dtype=np.float32))
        elif hasattr(entry, "boundaries"):
            boundaries.append(np.asarray(entry.boundaries, dtype=np.float32))
        elif isinstance(entry, dict) and "Boundaries" in entry:
            boundaries.append(np.asarray(entry["Boundaries"], dtype=np.float32))

    if not boundaries:
        raise ValueError(f"No boundaries found in: {mat_path}")

    stacked = np.stack(boundaries, axis=0)
    return np.mean(stacked, axis=0)

def compute_metrics(ground_truth_edges: np.ndarray, predicted_edges: np.ndarray) -> Tuple[float, float, float, float]:
    # ground truth edge is 0.0 to 1.0, predicted edge is 0.0 to 1.0
    ground_truth_edges = (ground_truth_edges > 0.2).astype(np.uint8)
    predicted_edges = (predicted_edges > 0.2).astype(np.uint8)
    precision = np.sum(np.logical_and(ground_truth_edges, predicted_edges)) / np.sum(predicted_edges)
    recall = np.sum(np.logical_and(ground_truth_edges, predicted_edges)) / np.sum(ground_truth_edges)
    F1 = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(np.logical_and(ground_truth_edges, predicted_edges)) / np.sum(np.logical_or(ground_truth_edges, predicted_edges))
    return {
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "accuracy": accuracy,
    }

def make_predictions(image_path: str, include_methods: List[str] = []):
    predictions = {}
    for method in include_methods:
        if method == "canny":
            predictions["canny"] = run_canny_edge(image_path, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        elif method == "sobel":
            predictions["sobel"] = run_sobel_edge(image_path)
        elif method == "roberts":
            predictions["roberts"] = run_roberts_edge(image_path)
        elif method == "prewitt":
            predictions["prewitt"] = run_prewitt_edge(image_path)
        elif method == "log":
            predictions["log"] = run_log_edge(image_path, LOG_THRESHOLD)
    return predictions
    
def compute_metrics_for_all_images(image_paths: List[str], ground_truth_paths: List[str], include_methods: str):
    metrics = {}
    for method in include_methods.split(","):
        metrics[method] = {
            "precision": [],
            "recall": [],
            "F1": [],
            "accuracy": [],
        }
    for image_path, ground_truth_path in tqdm(zip(image_paths, ground_truth_paths), total=len(image_paths)):
        ground_truth_edges = load_ground_truth_edges(ground_truth_path)
        predictions = make_predictions(image_path, include_methods.split(","))
        for method, prediction in predictions.items():
            results = compute_metrics(ground_truth_edges, prediction)
            for metric, value in results.items():
                metrics[method][metric].append(value)
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--include_methods", type=str, required=True)
    args = parser.parse_args()
    image_paths = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith(".jpg")][:50]
    print(len(image_paths))
    ground_truth_paths = [image_path.replace("images", "groundTruth").replace(".jpg", ".mat") for image_path in image_paths]
    metrics = compute_metrics_for_all_images(image_paths, ground_truth_paths, args.include_methods)
    # average the metrics for each method
    for method, method_metrics in metrics.items():
        for metric, values in method_metrics.items():
            metrics[method][metric] = np.mean(values)
    print(metrics)

if __name__ == "__main__":
    main()