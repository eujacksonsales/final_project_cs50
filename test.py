from cv_engine.processor import CVProcessor
import magic
import cv2
import matplotlib.pyplot as plt

cv_brain = CVProcessor()

# image, max_people = cv_brain.process_media("./image_test.jpg")

# plt.title("Deep Face + YOLO")
# plt.axis("off")
# plt.imshow(image[:,:,::-1])
# plt.show()

video_yolo_interval = cv_brain.process_media("./video_test.mp4")
video_yolo_without_interval = cv_brain.process_media2("./video_test.mp4")

# cap = cv2.VideoCapture("./output")

# if not cap.isOpened():
#     print("Error: Could not open video")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error occurred")
#         break

#     cv2.imshow("Frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
