import numpy as np
import math
import cv2
from typing import Literal
from curves import SpaceFillingCurve

MipmapStrategy = Literal["resize", "average"]

class FractalImageProcessor:
    def __init__(self, curve_strategy: SpaceFillingCurve, mipmap_strategy: MipmapStrategy = "resize"):
        self.curve = curve_strategy
        self.mipmap_strategy = mipmap_strategy

    def _pad_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
        """
        Pads image to the nearest power of 2.
        Returns padded image and original dimensions.
        """
        h, w, c = image.shape
        max_dim = max(h, w)
        order = math.ceil(math.log2(max_dim))
        size = 2 ** order
        
        average_color = np.mean(image, axis=(0, 1), dtype=np.uint8)
        padded = np.full((size, size, c), average_color, dtype=np.uint8)
        padded[:h, :w, :] = image
        
        return padded, (h, w), order

    def _generate_mipmaps_resize(self, image: np.ndarray, max_order: int) -> list[np.ndarray]:
        """
        Generates mipmaps by resizing the original image to each level.
        Uses cv2.resize with INTER_AREA interpolation for downsampling.
        Returns a list of images from smallest (1x1) to largest (full size).
        """
        mipmaps = []
        current = image
        # We store from highest resolution to lowest, then reverse
        mipmaps.append(current)
        
        for _ in range(max_order):
            h, w, _ = current.shape
            # Average pooling (resize by half)
            current = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            mipmaps.append(current)
            
        return list(reversed(mipmaps))

    def _generate_mipmaps_average(self, image: np.ndarray, max_order: int) -> list[np.ndarray]:
        """
        Generates mipmaps by averaging pixel colors along the curve from the previous iteration.
        Each pixel in a lower iteration is the average of 4 consecutive pixels (at curve positions 4t to 4t+3)
        from the higher iteration's curve ordering.
        Returns a list of images from smallest (1x1) to largest (full size).
        """
        mipmaps = []
        
        # Start by arranging the full resolution image according to the curve
        h, w, c = image.shape
        num_pixels = h * w
        
        # Flatten current level using the curve
        current_curve_data = np.zeros((num_pixels, 3), dtype=np.uint8)
        for d in range(num_pixels):
            x, y = self.curve.d_to_xy(d, max_order)
            current_curve_data[d] = image[y, x]
        
        # Now generate each lower level by averaging groups of 4 consecutive pixels
        for order in range(max_order, 0, -1):
            # Current order has 2^order x 2^order pixels
            # Next lower order will have 2^(order-1) x 2^(order-1) pixels
            lower_order = order - 1
            lower_size = 2 ** lower_order
            lower_num_pixels = lower_size * lower_size
            
            # Average every 4 consecutive pixels to create the lower level
            lower_curve_data = np.zeros((lower_num_pixels, 3), dtype=np.float32)
            for t in range(lower_num_pixels):
                # Average pixels at positions 4t, 4t+1, 4t+2, 4t+3
                lower_curve_data[t] = np.mean(current_curve_data[4*t:4*t+4], axis=0)
            
            # Convert back to image space for this level
            lower_image = np.zeros((lower_size, lower_size, c), dtype=np.uint8)
            for d in range(lower_num_pixels):
                x, y = self.curve.d_to_xy(d, lower_order)
                lower_image[y, x] = lower_curve_data[d].astype(np.uint8)
            
            mipmaps.append(lower_image)
            current_curve_data = lower_curve_data.astype(np.uint8)
        
        # mipmaps now contains from highest to lowest (excluding the original)
        # We need to reverse and add the original image at the end
        mipmaps.reverse()
        mipmaps.append(image)
        
        return mipmaps

    def _generate_mipmaps(self, image: np.ndarray, max_order: int) -> list[np.ndarray]:
        """
        Generates a pyramid of images (mipmaps) from 1x1 up to full resolution.
        Returns a list of images.
        """
        if self.mipmap_strategy == "resize":
            return self._generate_mipmaps_resize(image, max_order)
        elif self.mipmap_strategy == "average":
            return self._generate_mipmaps_average(image, max_order)
        else:
            raise ValueError(f"Unknown mipmap strategy: {self.mipmap_strategy}")

    def encode(self, image_input: str | np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Encodes image into a fractal stream.
        image_input: Path to image (str) or numpy array.
        Stream structure: [Level 0 data] [Level 1 data] ... [Level N data]
        Within each level, pixels are sorted by the Curve.
        """
        # Load and pad
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError("Image not found")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("Input must be a file path or numpy array")
            
        padded_img, original_dims, max_order = self._pad_image(img)
        
        # Generate pyramid (Level 0 is 1x1, Level max_order is full size)
        mipmaps = self._generate_mipmaps(padded_img, max_order)
        
        stream = []
        
        # Iterate through levels (BFS approach)
        for order, level_img in enumerate(mipmaps):
            h, w, _ = level_img.shape
            num_pixels = h * w
            
            # Flatten this level using the Curve
            level_data = np.zeros((num_pixels, 3), dtype=np.uint8)
            
            for d in range(num_pixels):
                x, y = self.curve.d_to_xy(d, order)
                level_data[d] = level_img[y, x] # Note: numpy uses y, x
            
            stream.append(level_data)
            
        # Concatenate all levels into one long 1D array
        full_stream = np.concatenate(stream)
        
        metadata = {
            'original_dims': original_dims,
            'max_order': max_order,
            'total_length': len(full_stream)
        }
        
        return full_stream, metadata

    def decode(self, stream: np.ndarray, metadata: dict, prefix_ratio: float = 1.0) -> np.ndarray:
        """
        Reconstructs image from a prefix of the stream.
        prefix_ratio: float between 0.0 and 1.0
        """
        max_order = metadata['max_order']
        orig_h, orig_w = metadata['original_dims']
        final_size = 2 ** max_order
        
        # Determine how much data we have
        total_len = metadata['total_length']
        available_len = int(total_len * prefix_ratio)
        current_stream = stream[:available_len]
        
        # Canvas for reconstruction (initially black)
        canvas = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        
        # Reconstruct level by level
        stream_ptr = 0
        
        for order in range(max_order + 1):
            level_pixels = (2**order) * (2**order)
            
            # Check if we have full, partial, or no data for this level
            remaining_stream = available_len - stream_ptr
            
            if remaining_stream <= 0:
                break
                
            points_to_read = min(level_pixels, remaining_stream)
            
            # Extract data for this level
            level_data = current_stream[stream_ptr : stream_ptr + points_to_read]
            
            # If we have full level data, we can just resize it to canvas
            # This acts as the "background" for higher details
            if points_to_read == level_pixels:
                # Reconstruct the small image
                temp_img = np.zeros((2**order, 2**order, 3), dtype=np.uint8)
                
                # Optimization: Vectorized mapping could be done here, 
                # but loop is clearer for the logic demonstration
                for d in range(points_to_read):
                    x, y = self.curve.d_to_xy(d, order)
                    temp_img[y, x] = level_data[d]
                
                # Upscale to final size (Nearest Neighbor or Linear)
                # This fills the "unknown" points of higher levels with current average
                scale_factor = final_size // (2**order)
                if scale_factor > 1:
                    upscaled = cv2.resize(temp_img, (final_size, final_size), interpolation=cv2.INTER_NEAREST)
                    canvas = upscaled
                else:
                    canvas = temp_img
            
            else:
                # Partial level data: Draw individual pixels on top of previous canvas
                scale_factor = final_size // (2**order)
                
                for d in range(points_to_read):
                    x_base, y_base = self.curve.d_to_xy(d, order)
                    color = level_data[d]
                    
                    # Calculate position in the final canvas
                    x_final = x_base * scale_factor
                    y_final = y_base * scale_factor
                    
                    # Fill the block corresponding to this pixel
                    canvas[y_final : y_final + scale_factor, 
                           x_final : x_final + scale_factor] = color
            
            stream_ptr += points_to_read

        # Crop padding
        final_image = canvas[:orig_h, :orig_w]
        return final_image
    