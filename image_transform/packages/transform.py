import cv2
import numpy as np

class ImageTransforms:
    def __init__(self, image_path: str):
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

    def resize(self, width: int, height: int):
        """Resize image to width x height."""
        return cv2.resize(self.image, (width, height))

    def rotate(self, angle: float):
        """Rotate image by angle (degrees)."""
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.image, matrix, (w, h))

    def flip_horizontal(self):
        """Flip image horizontally."""
        return cv2.flip(self.image, 1)

    def flip_vertical(self):
        """Flip image vertically."""
        return cv2.flip(self.image, 0)

    def to_gray(self):
        """Convert image to grayscale."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur(self, ksize: int = 5):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(self.image, (ksize, ksize), 0)

    def crop(self, x: int, y: int, w: int, h: int):
        """Crop image from (x,y) with width w and height h."""
        return self.image[y:y+h, x:x+w]

    def save(self, img, path: str):
        """Save processed image."""
        cv2.imwrite(path, img)