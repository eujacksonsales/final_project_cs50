from cv_engine.processor import CVProcessor
import magic
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

cv_brain = CVProcessor()

image, max_people = cv_brain.process_media("./image_test.jpg")

plt.title("Deep Face + YOLO")
plt.axis("off")
plt.imshow(image[:,:,::-1])
plt.text(x=10, y=10, s=max_people)
plt.show()

