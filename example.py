import numpy as np
import cv2
import os
from processor import FractalImageProcessor
from curves import HilbertCurve, ZOrderCurve
from metrics import calculate_mse, calculate_psnr

if __name__ == "__main__":
    # Example usage
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup
    # Create a dummy image for demonstration (gradient)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            dummy_img[i, j] = [i * 2, j * 2, (i+j)]
    input_path = os.path.join(output_dir, "test_input.png")
    cv2.imwrite(input_path, dummy_img)
    
    # 2. Initialize Processor with Hilbert Curve
    hilbert_proc = FractalImageProcessor(HilbertCurve())
    
    # 3. Encode
    print("Encoding image...")
    stream, meta = hilbert_proc.encode(input_path)
    print(f"Stream length: {len(stream)} pixels")
    
    # 4. Decode with different prefixes
    ratios = [0.01, 0.1, 0.5, 1.0] # 1%, 10%, 50%, 100%
    
    for r in ratios:
        print(f"Decoding with {r*100}% of data...")
        recon_img = hilbert_proc.decode(stream, meta, prefix_ratio=r)
        
        # Calculate Error
        original = cv2.imread(input_path)
        mse = calculate_mse(original, recon_img)
        psnr = calculate_psnr(original, recon_img)
        print(f"MSE Error: {mse:.2f}, PSNR Error: {psnr:.2f}")
        
        # Save result
        cv2.imwrite(os.path.join(output_dir, f"output_hilbert_{int(r*100)}.png"), recon_img)

    # 5. Compare with Z-Order Curve
    print("\n--- Comparing with Z-Order ---")
    z_proc = FractalImageProcessor(ZOrderCurve())
    stream_z, meta_z = z_proc.encode(input_path)
    recon_z = z_proc.decode(stream_z, meta_z, prefix_ratio=0.1)
    mse_z = calculate_mse(cv2.imread(input_path), recon_z)
    psnr_z = calculate_psnr(cv2.imread(input_path), recon_z)
    print(f"Z-Order at 10%: MSE: {mse_z:.2f}, PSNR: {psnr_z:.2f}")
