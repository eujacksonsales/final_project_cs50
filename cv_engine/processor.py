# cv_engine/processor.py
import cv2
import numpy as np
import os
import magic
from deepface import DeepFace
import matplotlib.pyplot as plt


class CVProcessor:
    def __init__(self, yolo_interval = 5, deep_face_interval = 150):

        # TODO: Initialize your models here (YOLO, DeepFace) <---- AI Help
        
        try:
            from ultralytics import YOLO
            import torch
            self.yolo_model = YOLO("yolov8n.pt")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model.to(self.device)
            print(f"YOLOv8n run and using: {self.device}")
        except Exception as e:
            print("Error the YOLOv8 failed to load", e)
            self.yolo_model = None

        self.frame_idx = 0
        self.max_number_people = 0
        self.cached_results_face = []
        self.DEEP_FACE_INTERVAL = deep_face_interval

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

        # TODO: 
        # 1. Load the image/video using OpenCV
        # 2. Run YOLO (every 5 frames if video)
        # 3. Run DeepFace based on 'options'
        # 4. Draw bounding boxes
        # 5. Save the new file to 'static/output/'

        print(f"Processing {file_path} with options: {options}")

        text_person = ""
        # If the file is image 
        type_file = self.get_file_type(file_path)

        if type_file == "image":
            #Run image
            image = cv2.imread(file_path)
            #Run YOLO
            results_yolo = self.yolo_model.track(source=image, classes=[0] , tracker="bytetrack.yaml", conf=0.5)
                                
            for result in results_yolo:
                boxes = result.boxes
                new_results_face = []
                for detection in boxes:

                    # Get area of person
                    x1, y1, x2, y2 = map(int, detection.xyxy.cpu().numpy()[0])
                    #Extract confidence
                    conf = detection.conf.cpu().item()

                    #Track id
                    track_id = int(detection.id.cpu().item()) if detection.id is not None else -1
                    
                    # Get the max number of people in the video
                    if self.max_number_people < track_id:
                        self.max_number_people = track_id
                    
                    # Crop the person image for emotion analyze
                    person = image[y1:y2, x1:x2]

                    results_deep_face = DeepFace.analyze(person, actions=options, enforce_detection=False, detector_backend="retinaface")

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

                    text_person = f"P: {track_id}; Conf: {conf:.2f}; G: {gender}; \n A: {age}; E: {emotion}; Et: {ethnicity}"
                        
                    # Draw bounding box face
                    cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

                    # Draw bounding box person
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text for each person
                    cv2.putText(image, text_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            return (image, self.max_number_people)

        elif type_file == "video":

            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print("The video not run")

            results_yolo = self.yolo_model.track(source=file_path, classes=[0] , tracker="bytetrack.yaml", conf=0.5, stream=True, persist=True)
                                
            for frame_result in results_yolo:
                self.frame_idx +=1
                boxes = frame_result.boxes
                new_results_face = []
                for detection in boxes:

                    # Get area of person
                    x1, y1, x2, y2 = map(int, detection.xyxy.cpu().numpy()[0])
                    #Extract confidence
                    conf = detection.conf.cpu().item()

                    #Track id
                    track_id = int(detection.id.cpu().item()) if detection.id is not None else -1
                    
                    # Get the max number of people in the video
                    if self.max_number_people < track_id:
                        self.max_number_people = track_id
                    
                    # Crop the person image for emotion analyze
                    person = image[y1:y2, x1:x2]


                    if self.frame_idx % self.DEEP_FACE_INTERVAL == 0 or self.frame_idx == 5:
                        
                        results_deep_face = DeepFace.analyze(person, actions=options, enforce_detection=False, detector_backend="retinaface")
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

                        text_person = f"P: {track_id}; Conf: {conf:.2f}; G: {gender}; A: {age}; E: {emotion}; Et: {ethnicity}"
                        
                        # Draw bounding box face
                        cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

                        new_results_face.append({"id_people": track_id, "conf": conf, "gender": gender, "age": age, "emotion": emotion,
                                                            "ethnicity": ethnicity, "face_x1" : face_x1,"face_y1": face_y1, "face_x2": face_x2, "face_y2": face_y2})
                        
                        # Cache the result 
                        self.cached_results_face = new_results_face
                                        # Draw text with details
                    
                    # Draw bounding box person
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


                if self.frame_idx % self.DEEP_FACE_INTERVAL != 0 and self.frame_idx != 5:
                    for face in self.cached_results_face:
                        gender = face["gender"]
                        emotion = face["emotion"]
                        age = face["age"]
                        ethnicity = face["ethnicity"]

                        face_x1 = face["face_x1"]
                        face_y1 = face["face_y1"]
                        face_x2 = face["face_x2"]
                        face_y2 = face["face_y2"]

                        text_person = f"P: {face['id_people']}; Conf: {face['conf']:.2f}; G: {gender}; A: {age}; E: {emotion}; Et: {ethnicity}"
                        
                        # Draw bounding box face
                        cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
                
                cv2.putText(image, text_person, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)


            #Run video
            print()
        else:
            print("Error")
        


        # MOCK RETURN (For testing the web part before the CV part is ready)
        # return {
        #     "processed_file": "output/placeholder.jpg", # This will be relative to /static
        #     "stats": {
        #         "people_count": 0,
        #         "gender_summary": "Not implemented",
        #         "detected_emotions": []
        #     }
        # }
    

