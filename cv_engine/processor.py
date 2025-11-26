# cv_engine/processor.py
import cv2
import numpy as np
import os
import magic
from deepface import DeepFace
import matplotlib.pyplot as plt


class CVProcessor:
    def __init__(self):

        # TODO: Initialize your models here (YOLO, DeepFace) <---- AI Help
        
        try:
            from ultralytics import YOLO
            import torch
            self.yolo_model = YOLO("yolov8n.pt")
            print("YOLOv8n run")
        except Exception as e:
            print("Error the YOLOv8 failed to load", e)
            self.yolo_model = None


        # This runs once when the server starts, so we don't reload models every request. <---- AI Help
        pass

    def get_file_type(self, filepath):
        try:
            mime_type = magic.from_file(filepath, mime=True)
            if mime_type.startswith("image/"):
                return "image"
            elif mime_type.startswith("video/"):
                return "video"
            else:
                return "other"
        except Exception as e:
            print("Error: ", e)

    def process_media(self, file_path: str, options: tuple = ('age', 'gender', 'emotion', 'race')):
        """
        Args:
            file_path: Path to the uploaded image/video.
            options: Dictionary of booleans (e.g., {'count_people': True, 'emotion_joy': False})
        
        Returns:
            Dictionary containing the path to the processed file and the text summary.
        """
        print(f"Processing {file_path} with options: {options}")

        list = []
        # If the file is image 
        type_file = self.get_file_type(file_path)

        if type_file == "image":
            #Run image
            countPeople = 0
            image = cv2.imread(file_path)
            image_original = image.copy()
            width = image.shape[1]
            height = image.shape[0]
            #Run YOLO
            results_yolo = self.yolo_model.predict(image, classes = [0])

            for result in results_yolo:
                boxes = result.boxes
                for detection in boxes:
                    countPeople += 1

                    x1, y1, x2, y2 = map(int, detection.xyxy[0])

                    person = image[y1:y2, x1:x2]
                    text_person = ""

                    results_deep_face = DeepFace.analyze(person, actions=options, enforce_detection=False, detector_backend="retinaface")

                    if results_deep_face is not None:
                        gender = results_deep_face[0]["dominant_gender"]
                        emotion = results_deep_face[0]["dominant_emotion"]
                        age = results_deep_face[0]["age"]
                        ethnicity = results_deep_face[0]["dominant_race"]

                        # DeepFace relative face box
                        fx1 = results_deep_face[0]["region"]["x"]
                        fy1 = results_deep_face[0]["region"]["y"]
                        fw = results_deep_face[0]["region"]["w"]
                        fh = results_deep_face[0]["region"]["h"]

                        face_x1 = x1 + fx1
                        face_y1 = y1 + fy1
                        face_x2 = x1 + fx1 + fw
                        face_y2 = y1 + fy1 + fh

                        text_person = f"P: {countPeople}; G: {gender}; A: {age}; E: {emotion}; Et: {ethnicity}"
                        
                        # Draw bounding box face
                        cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

                    list.append(text_person)
                    # Draw text with details
                    cv2.putText(image, text_person, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    
                    # Draw bounding box person
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            


            return image

        elif type_file == "video":
            #Run video
            print()
        else:
            print("Error")
        
        # TODO: 
        # 1. Load the image/video using OpenCV
        # 2. Run YOLO (every 5 frames if video)
        # 3. Run DeepFace based on 'options'
        # 4. Draw bounding boxes
        # 5. Save the new file to 'static/output/'
        


        # MOCK RETURN (For testing the web part before the CV part is ready)
        # return {
        #     "processed_file": "output/placeholder.jpg", # This will be relative to /static
        #     "stats": {
        #         "people_count": 0,
        #         "gender_summary": "Not implemented",
        #         "detected_emotions": []
        #     }
        # }
    

