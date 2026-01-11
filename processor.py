import numpy as np
import math
import cv2
from curves import SpaceFillingCurve

class FractalImageProcessor:
    def __init__(self, curve_strategy: SpaceFillingCurve):
        self.curve = curve_strategy

    def _pad_image(self, image):
        """
        Pads image to the nearest power of 2.
        Returns padded image and original dimensions.
        """
        h, w, c = image.shape
        max_dim = max(h, w)
        order = math.ceil(math.log2(max_dim))
        size = 2 ** order
        
        padded = np.zeros((size, size, c), dtype=np.uint8)
        padded[:h, :w, :] = image
        
        return padded, (h, w), order

    def _generate_mipmaps(self, image, max_order):
        """
        Generates a pyramid of images (mipmaps) from 1x1 up to full resolution.
        Returns a list of images.
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

    def encode(self, image_input):
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

    def decode(self, stream, metadata, prefix_ratio=1.0):
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
    