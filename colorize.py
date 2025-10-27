import numpy as np
import cv2
import os

MODEL_DIR = "model"
PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")

print("🔹 Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(input_image):
    """Takes a grayscale or BGR image (NumPy) and returns colorized BGR image"""
    if input_image is None:
        raise ValueError("Invalid image")

    image = input_image.astype("float32") / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")
