import numpy as np


def vertical_gradient(size: int) -> np.ndarray:
    """Generate a vertical gradient from black (top) to white (bottom)."""
    gradient = np.linspace(0, 255, size, dtype=np.uint8)
    image = np.repeat(gradient[:, np.newaxis], size, axis=1)
    return np.stack([image, image, image], axis=2)


def horizontal_gradient(size: int) -> np.ndarray:
    """Generate a horizontal gradient from black (left) to white (right)."""
    gradient = np.linspace(0, 255, size, dtype=np.uint8)
    image = np.repeat(gradient[np.newaxis, :], size, axis=0)
    return np.stack([image, image, image], axis=2)


def diagonal_gradient(size: int) -> np.ndarray:
    """Generate a diagonal gradient from top-left to bottom-right."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    image = ((xx + yy) / 2 * 255).astype(np.uint8)
    return np.stack([image, image, image], axis=2)


def random_noise(size: int) -> np.ndarray:
    """Generate random noise."""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def checkerboard(size: int, squares: int = 8) -> np.ndarray:
    """Generate a checkerboard pattern."""
    square_size = size // squares
    image = np.zeros((size, size), dtype=np.uint8)
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                image[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255
    return np.stack([image, image, image], axis=2)


def horizontal_stripes(size: int, num_stripes: int = 8) -> np.ndarray:
    """Generate horizontal stripes alternating black and white."""
    stripe_height = size // num_stripes
    image = np.zeros((size, size), dtype=np.uint8)
    for i in range(num_stripes):
        if i % 2 == 0:
            image[i*stripe_height:(i+1)*stripe_height, :] = 255
    return np.stack([image, image, image], axis=2)


def vertical_stripes(size: int, num_stripes: int = 8) -> np.ndarray:
    """Generate vertical stripes alternating black and white."""
    stripe_width = size // num_stripes
    image = np.zeros((size, size), dtype=np.uint8)
    for i in range(num_stripes):
        if i % 2 == 0:
            image[:, i*stripe_width:(i+1)*stripe_width] = 255
    return np.stack([image, image, image], axis=2)


def diagonal_stripes(size: int, num_stripes: int = 8) -> np.ndarray:
    """Generate diagonal stripes from top-left to bottom-right."""
    image = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if ((i + j) // (size // num_stripes)) % 2 == 0:
                image[i, j] = 255
    return np.stack([image, image, image], axis=2)


def solid_color(size: int, r: int = 128, g: int = 128, b: int = 128) -> np.ndarray:
    """Generate a solid color image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b
    return image


def concentric_circles(size: int, num_circles: int = 8) -> np.ndarray:
    """Generate concentric circles alternating black and white."""
    center = size / 2
    max_radius = np.sqrt(2) * center
    image = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            ring = int(distance / max_radius * num_circles)
            if ring % 2 == 0:
                image[i, j] = 255
    
    return np.stack([image, image, image], axis=2)


def radial_gradient(size: int) -> np.ndarray:
    """Generate a radial gradient from center (white) to edges (black)."""
    center = size / 2
    max_distance = np.sqrt(2) * center
    image = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            value = int((1 - distance / max_distance) * 255)
            image[i, j] = max(0, min(255, value))
    
    return np.stack([image, image, image], axis=2)


def color_gradient(size: int) -> np.ndarray:
    """Generate a colorful gradient (red to green to blue)."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Red channel: high on left, low on right
    gradient = np.linspace(255, 0, size, dtype=np.uint8)
    image[:, :, 0] = np.repeat(gradient[np.newaxis, :], size, axis=0)
    
    # Green channel: high at top, low at bottom
    gradient = np.linspace(255, 0, size, dtype=np.uint8)
    image[:, :, 1] = np.repeat(gradient[:, np.newaxis], size, axis=1)
    
    # Blue channel: increases diagonally
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    image[:, :, 2] = ((xx + yy) / 2 * 255).astype(np.uint8)
    
    return image


def random_images(num_images: int, size: int, seed: int | None = None) -> list[np.ndarray]:
    """Generate a list of random images."""
    generator = np.random.default_rng(seed)
    return [random_image(generator, size) for _ in range(num_images)]
    
    
def random_image(generator: np.random.Generator, size: int) -> np.ndarray:
    """Generate a random image."""
    index: int = generator.integers(0, 12)
    match index:
        case 0:
            squares = generator.integers(2, 10)
            return checkerboard(size, squares)
        case 1:
            stripes = generator.integers(2, 10)
            return horizontal_stripes(size, stripes)
        case 2:
            stripes = generator.integers(2, 10)
            return vertical_stripes(size, stripes)
        case 3:
            stripes = generator.integers(2, 10)
            return diagonal_stripes(size, stripes)
        case 4:
            return solid_color(size)
        case 5:
            circles = generator.integers(2, 10)
            return concentric_circles(size, circles)
        case 6:
            return radial_gradient(size)
        case 7:
            return color_gradient(size)
        case 8:
            return random_noise(size)
        case 9:
            return vertical_gradient(size)
        case 10:
            return horizontal_gradient(size)
        case 11:
            return diagonal_gradient(size)
        case _:
            raise ValueError(f"Invalid random image type index: {index}")