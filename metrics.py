import numpy as np
import math

def calculate_mse(img1, img2):
    """Calculates Mean Squared Error between two images."""
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return err

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))