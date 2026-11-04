"""Second metric: Laplacian Variance (Blur/Sharpness detection)."""
import numpy as np

def compute_laplacian_variance(image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    kernel = np.array([[0, 1, 0], 
                       [1, -4, 1], 
                       [0, 1, 0]])
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    laplacian = (kernel[0,1] * gray[:-2, 1:-1] +
                 kernel[1,0] * gray[1:-1, :-2] +
                 kernel[1,1] * gray[1:-1, 1:-1] +
                 kernel[1,2] * gray[1:-1, 2:] +
                 kernel[2,1] * gray[2:, 1:-1])
    return float(np.var(laplacian))
