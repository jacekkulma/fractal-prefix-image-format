import numpy as np

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
