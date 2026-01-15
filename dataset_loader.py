import random
import numpy as np
from datasets import load_dataset


DATASET_NAME = "timm/oxford-iiit-pet"


def load_images(num_samples: int | None = None, random_seed: int | None = None) -> list[np.ndarray]:
    """
    Loads images from a dataset and returns them as a list of numpy arrays.
    
    Args:
        num_samples: Optional number of samples to load. If None, loads all images.
        random_seed: Optional random seed for reproducible sampling.
    
    Returns:
        List of numpy arrays, where each array represents an image.
        Each array has shape (height, width, channels) with dtype uint8.
    """
    dataset = load_dataset(DATASET_NAME)
    
    images = []
    for example in dataset["train"]:
        image = example["image"]
        img_array = np.array(image, dtype=np.uint8)
        
        if len(img_array.shape) != 3:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
        
        images.append(img_array)

    original_length = len(images)

    if random_seed is not None:
        generator = random.Random(random_seed)
        generator.shuffle(images)

    if num_samples is not None:
        images = images[:num_samples]
    
    print(f"Loaded {len(images)}/{original_length} images")
    
    return images

