import numpy as np

from curves import SpaceFillingCurve
from processor import FractalImageProcessor


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Mean Squared Error between two images."""
    val_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    if val_range == 0:
        return 0

    normalized_img1 = (img1 - np.min(img1)) / val_range
    normalized_img2 = (img2 - np.min(img2)) / val_range
    err = np.sum((normalized_img1 - normalized_img2) ** 2)
    err /= float(normalized_img1.shape[0] * normalized_img1.shape[1] * normalized_img1.shape[2])
    return err


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_error_curve(
    processor: FractalImageProcessor,
    images: list[np.ndarray],
    prefix_ratios: list[float],
) -> np.ndarray:
    errors = np.zeros((len(images), len(prefix_ratios)))
    for i, image in enumerate(images):
        encoded, metadata = processor.encode(image)
        for j, prefix_ratio in enumerate(prefix_ratios):
            decoded = processor.decode(encoded, metadata, prefix_ratio)
            error = calculate_mse(image, decoded)
            errors[i, j] = error

    return np.mean(errors, axis=0)


def area_under_curve(curve: np.ndarray, prefix_ratios: list[float]) -> float:
    if len(curve) != len(prefix_ratios):
        raise ValueError(f"Curve and prefix ratios must have the same length - found {len(curve)} and {len(prefix_ratios)}")
    return np.trapezoid(curve, prefix_ratios)


def calculate_curve_locality(
    curve: SpaceFillingCurve,
    order: int,
    n_samples: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates locality metric for a space-filling curve.
    
    Measures how well the curve preserves locality: if two points are close
    in parameter space (t values), are they also close in 2D space?
    
    Args:
        curve: SpaceFillingCurve instance to analyze
        order: Order of the curve (grid size is 2^order x 2^order)
        n_samples: Number of random t values to sample from [0, 1]
        n_pairs: Number of pairs to sample (None = all pairs)
    
    Returns:
        t_differences: Array of differences in t values
        xy_distances: Array of normalized Euclidean distances in x/y space
        Both arrays are sorted by t_differences
    """
    if n_samples < 5:
        raise ValueError(f"n_samples must be at least 5, found {n_samples}")

    n_pairs = n_samples * 2

    generator = np.random.default_rng(seed)

    # Generate random t values from [0, 1]
    t_values = generator.random(n_samples)
    t_values.sort()  # Sort for easier processing
    
    # Convert t values to distances and then to x/y coordinates
    max_distance = curve.get_max_distance(order)
    grid_size = 2 ** order
    distances = (t_values * (max_distance - 1)).astype(int)
    coordinates = np.array([curve.d_to_xy(d, order) for d in distances], dtype=float)
    
    # Normalize coordinates to [0, 1]
    coordinates = coordinates / (grid_size - 1)
    
    # Compute all pairs
    t_diffs = []
    xy_dists = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            t_diff = t_values[j] - t_values[i]
            xy_dist = np.linalg.norm(coordinates[i] - coordinates[j])
            t_diffs.append(t_diff)
            xy_dists.append(xy_dist)
    
    # Convert to arrays
    t_diffs = np.array(t_diffs)
    xy_dists = np.array(xy_dists)
    
    indices = generator.choice(len(t_diffs), n_pairs, replace=False)
    t_diffs = t_diffs[indices]
    xy_dists = xy_dists[indices]
    
    # Sort by t differences
    sort_indices = np.argsort(t_diffs)
    t_diffs = t_diffs[sort_indices]
    xy_dists = xy_dists[sort_indices]
    
    return t_diffs, xy_dists
