import numpy as np
import cv2
import os
from processor import FractalImageProcessor
from curves import HilbertCurve, ZOrderCurve, ScanlineCurve
from metrics import calculate_mse, calculate_psnr

if __name__ == "__main__":
    # Example usage
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup
    # Create a complex dummy image: Top half solid, Bottom half random noise
    # This highlights differences: Scanline gets stuck on the easy top part.
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Top half: Solid Gray (Easy to compress, low-res mipmap handles it well)
    dummy_img[:50, :] = 128
    # Bottom half: Random Noise (Hard to compress, needs pixel data)
    dummy_img[50:, :] = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
    
    input_path = os.path.join(output_dir, "test_input.png")
    cv2.imwrite(input_path, dummy_img)
    
    # Load original once for metrics
    original_img = cv2.imread(input_path)
    
    strategies = [
        ("Hilbert", HilbertCurve()),
        ("Z-Order", ZOrderCurve()),
        ("Scanline", ScanlineCurve())
    ]
    
    ratios = [0.01, 0.1, 0.5, 1.0]
    
    for name, curve in strategies:
        print(f"\n--- Processing with {name} ---")
        proc = FractalImageProcessor(curve)
        stream, meta = proc.encode(input_path)
        
        for r in ratios:
            recon = proc.decode(stream, meta, prefix_ratio=r)
            mse = calculate_mse(original_img, recon)
            psnr = calculate_psnr(original_img, recon)
            print(f"Decoding with {r*100:>5.1f}% data -> MSE: {mse:>7.2f}, PSNR: {psnr:>6.2f}")
            
            cv2.imwrite(os.path.join(output_dir, f"output_{name.lower()}_{int(r*100)}.png"), recon)
