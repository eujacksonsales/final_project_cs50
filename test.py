from cv_engine.processor import CVProcessor
import magic
import cv2
import matplotlib.pyplot as plt

cv_brain = CVProcessor()

#text_api = cv_brain.process_media("./image_test.jpg")



# plt.title("Deep Face + YOLO")
# plt.axis("off")
# plt.imshow(image[:,:,::-1])
# plt.show()

video_yolo_interval = cv_brain.process_media("./video_test.mp4")
print(video_yolo_interval)
