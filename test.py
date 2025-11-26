from cv_engine.processor import CVProcessor
import magic
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

cv_brain = CVProcessor()

image = cv_brain.process_media("./image_test.jpg")

plt.title("Deep Face + YOLO")
plt.axis("off")
plt.show(image[:,:,::-1])


