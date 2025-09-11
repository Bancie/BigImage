import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

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

    def am_ban(self):
        """
        Create a negative (âm bản) version of the image.
        """
        negative = cv2.bitwise_not(self.image)
        return negative
    
    def histogram(self):
        """
        Display the histogram of the image for each channel.
        """
        color = ('b', 'g', 'r')
        plt.figure(figsize=(10, 5))
        for i, col in enumerate(color):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram of Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def tru_anh(self, other_image_path: str):
        """
        Subtract another image from the current image.
        Images must be the same size.
        """
        other = cv2.imread(other_image_path)
        if other is None:
            raise ValueError(f"Could not read image: {other_image_path}")

        # Resize if needed
        if other.shape != self.image.shape:
            other = cv2.resize(other, (self.image.shape[1], self.image.shape[0]))

        result = cv2.subtract(self.image, other)
        return result

    def tach_bit_planes(self):
        """
        Tách 8 mặt phẳng bit (bit planes) của ảnh xám.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        bit_planes = []

        plt.figure(figsize=(12, 6))
        for i in range(8):
            bit_plane = cv2.bitwise_and(gray, (1 << i))
            bit_plane = np.where(bit_plane > 0, 255, 0).astype(np.uint8)
            bit_planes.append(bit_plane)

            plt.subplot(2, 4, i+1)
            plt.imshow(bit_plane, cmap='gray')
            plt.title(f'Bit Plane {i}')
            plt.axis('off')

        plt.suptitle('8-Bit Planes')
        plt.tight_layout()
        plt.show()
        return bit_planes

    def to_3d(self):
        """Convert the image to a 3D surface plot (grayscale)."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Get dimensions
        h, w = gray.shape
        X = np.arange(0, w)
        Y = np.arange(0, h)
        X, Y = np.meshgrid(X, Y)
        Z = gray

        # Plot 3D surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("2D Image to 3D Surface")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Intensity")
        plt.show()
    
    def detect_objects(self, model_path: str, config_path: str = None, conf_threshold: float = 0.5):
        """
        Detect objects with OpenCV DNN.
        Supports:
          - Caffe:  model: *.caffemodel, config: *.prototxt
          - ONNX:   model: *.onnx (config_path ignored)
        """

        # Resolve absolute paths and validate
        model_path = os.path.abspath(model_path) if model_path else None
        config_path = os.path.abspath(config_path) if config_path else None

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        net = None
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".onnx":
            # ONNX
            net = cv2.dnn.readNet(model_path)
            input_size = (300, 300)  # adjust if your ONNX expects other size
            scalefactor = 1.0 / 127.5
            mean = (127.5, 127.5, 127.5)
            swap_rb = True
        elif ext == ".caffemodel":
            # Caffe requires a .prototxt
            if not config_path or not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Caffe config (.prototxt) not found: {config_path}"
                )
            # IMPORTANT: readNetFromCaffe(prototxt, caffemodel)
            net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            # MobileNet-SSD (Caffe) preprocessing
            input_size = (300, 300)
            scalefactor = 0.007843  # 1/127.5
            mean = (127.5, 127.5, 127.5)
            swap_rb = False
        else:
            raise ValueError(
                f"Unsupported model extension '{ext}'. Use .onnx or (.caffemodel + .prototxt)."
            )

        # Prepare blob and inference
        h, w = self.image.shape[:2]
        blob = cv2.dnn.blobFromImage(self.image, scalefactor, input_size, mean, swapRB=swap_rb, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # MobileNet-SSD (VOC) labels
        CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

        output = self.image.copy()
        if detections.ndim != 4 or detections.shape[0] == 0:
            return output  # nothing to draw, return original

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            # Guard against out-of-range class id
            label_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else str(class_id)

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)
            startX, startY, endX, endY = box.astype(int)

            cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(output, f"{label_name}: {confidence*100:.1f}%",
                        (startX, max(0, startY - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        return output