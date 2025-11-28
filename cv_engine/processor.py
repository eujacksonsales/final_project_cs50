# cv_engine/processor.py
import cv2, time
import numpy as np
import os
import magic
from openvino.runtime import Core
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion", "FP16", "emotions-recognition-retail-0003.xml")

class CVProcessor:
    def __init__(self, insight_face_interval = 10, yolo_interval = 5, use_gpu = True):

        # TODO: Initialize your models here (YOLO, ONNX Emotion) <---- AI Help
        print("Loading models...")
        
        # 1. Load YOLO
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_model.to(self.device)
            print(f"YOLOv8 running on: {self.device}")
        except Exception as e:
            print(f"YOLO Error: {e}")

        # 2. Load OpenVINO Emotion
        try:
            self.emotion_core = Core()
            model_xml = EMOTION_MODEL_PATH
            # Automatically find the .bin file if it shares the name
            self.compiled_emotion = self.emotion_core.compile_model(model_xml, "CPU") 
            self.emotion_input = self.compiled_emotion.input(0)
            self.emotion_output = self.compiled_emotion.output(0)
        except Exception as e:
            print(f"OpenVINO Error: {e}")

        # 3. Load InsightFace
        # ctx_id=0 for GPU, -1 for CPU
        ctx = 0 if self.device == 'cuda' else -1
        self.app_insightface = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'] if ctx==0 else ['CPUExecutionProvider'])
        self.app_insightface.prepare(ctx_id=ctx, det_size=(640, 640))

        self.INSIGHT_FACE_INTERVAL = insight_face_interval
        self.YOLO_INTERVAL = yolo_interval

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

    def process_media(self, file_path: str, options: dict):
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

        #Temp variables
        persons_stats = {}
        tracked_people = set()


        # If the file is image 
        type_file = self.get_file_type(file_path)

        if type_file == "image":
            return self.process_image(file_path=file_path, options=options, persons_stats=persons_stats, tracked_people=tracked_people)
        elif type_file == "video":
            return self.process_video(file_path=file_path, options=options, persons_stats=persons_stats, tracked_people=tracked_people)
        else:
            print("Error")
        
    
    # AI Helps 
    def analyze_emotion(self, face_crop):
        face_img = cv2.resize(face_crop, (64,64))
        face_img = face_img.transpose(2,0,1)[None, ...].astype(np.float32)

        result = self.compiled_emotion({self.emotion_input.any_name: face_img})[self.emotion_output]
        emotion_id = int(np.argmax(result))

        label = ["neutral", "happy", "sad", "surprise", "anger"]
        return label[emotion_id]
        
    def process_video(self, file_path, options, persons_stats, tracked_people):
        cached_results_face = []
        cached_results_yolo = []
        frame_idx = 0
        
        cap = cv2.VideoCapture(file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path_video = os.path.join(BASE_DIR, "static", "output", "output.mp4")
        output_video = cv2.VideoWriter(path_video, fourcc, fps, (frame_width, frame_height)) 
        
        if not cap.isOpened():
            print("The video not run")

        start_time = time.time()                   
        while cap.isOpened():
            
            ret, frame = cap.read()

            if ret:
                frame_idx +=1              

                if frame_idx % self.YOLO_INTERVAL == 0:
                    
                    results = self.yolo_model.track(source=frame, classes=[0] , tracker="bytetrack.yaml", conf=0.5, stream=True, persist=True)
                    new_results_yolo = []
                    for result in results:
                        if result.boxes and result.boxes.is_track:
                            
                            # All persons boxes
                            boxes = result.boxes 

                            new_results_face = []
                            for detection in boxes:
                                # Get area of person resized
                                x1, y1, x2, y2 = map(int, detection.xyxy.cpu().numpy()[0])

                                #Extract confidence
                                conf = detection.conf.cpu().item()

                                #Track id
                                track_id = int(detection.id.cpu().item()) if detection.id is not None else -1
                                
                                # Get the max number of people in the video
                                tracked_people.add(track_id)
                                parts_text = [f"P:{track_id}", f"C:{conf:.2f}"]
                                if options["gender"] or options["age"] or any(options["emotions"].values()):
                                    # Crop the person image for emotion analyze
                                    person = frame[y1:y2, x1:x2]

                                    
                                    if frame_idx % self.INSIGHT_FACE_INTERVAL == 0:
                                        
                                        face = self.app_insightface.get(person)


                                        if len(face) != 0:                    
                                            fx1, fy1, fx2, fy2 = map(int, face[0].bbox)

                                            # <--- AI HELP to fix the negative numbers out the image person crop --->
                                            h, w = person.shape[:2]

                                            # Clip negative and out-of-range coordinates
                                            fx1 = max(0, min(fx1, w - 1))
                                            fy1 = max(0, min(fy1, h - 1))
                                            fx2 = max(0, min(fx2, w - 1))
                                            fy2 = max(0, min(fy2, h - 1))
                                            
                                            # After clipping, ensure valid ordering
                                            if fx2 <= fx1 or fy2 <= fy1:
                                                print("Invalid face bbox after clipping:", fx1, fy1, fx2, fy2)
                                                pass  # skip this face

                                            else:
                                                #Analyze only the face
                                                face_crop = person[fy1:fy2, fx1:fx2]
                                                if face_crop.size == 0:
                                                    print("Empty crop: ", fx1, fy1, fx2, fy2, person.shape)
                                                    pass
                                                # <------------------------------------------>
                                                else:

                                                    if options["gender"]:
                                                        gender = "male" if face[0].gender == 1 else "female"
                                                        parts_text.append(f"G:{gender}")
                                                    else:
                                                        gender = None
                                                    
                                                    if options["age"]:
                                                        age = face[0].age
                                                        parts_text.append(f"G:{age}")
                                                    else:
                                                        age = None

                                                    #Analyze only the face
                                                    if any(options["emotions"].values()):
                                                        emotion = self.analyze_emotion(person[fy1:fy2, fx1:fx2])
                                                    else:
                                                        emotion = None

                                                    if options["emotions"][emotion]:
                                                        parts_text.append(f"G:{emotion}")

                                                    # Face relative to entire image
                                                    face_x1 = x1 + fx1
                                                    face_y1 = y1 + fy1
                                                    face_x2 = x1 + fx2
                                                    face_y2 = y1 + fy2

                                                    #<---- AI Help ---->
                                                    text_person = "; ".join(parts_text) + ";"
                                                    #<----------------->
                                                    
                                                    # Draw bounding box face
                                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

                                                    new_results_face.append({"id_people": track_id, "conf": conf, "gender": gender, "age": age, "emotion": emotion,
                                                                            "face_x1" : face_x1,"face_y1": face_y1, "face_x2": face_x2, "face_y2": face_y2,  "x1": x1, "y1": y1})
                                                    
                                                    # Put text details in face of people
                                                    cv2.putText(frame, text_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                                    # Cache the result 
                                                    cached_results_face = new_results_face

                                                    #<--- AI Help to make possible register a list of emotions into a dict of dicts
                                                    if track_id not in persons_stats:
                                                        persons_stats[track_id] = {
                                                            "confidence": conf, "gender": gender, "age": age, "emotions": []
                                                        }
                                                        if options["emotions"][emotion]:
                                                            persons_stats[track_id]["emotions"].append(emotion)
                                                    else:
                                                        if options["emotions"][emotion]:
                                                            persons_stats[track_id]["emotions"].append(emotion)
                                # Draw bounding box person
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                new_results_yolo.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                                cached_results_yolo = new_results_yolo                                    

                else:
                    for person in cached_results_yolo:
                        x1 = person["x1"]
                        y1 = person["y1"]
                        x2 = person["x2"]
                        y2 = person["y2"]

                        # Draw bounding box person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if frame_idx % self.INSIGHT_FACE_INTERVAL != 0:
                    for face in cached_results_face:
                        x1 = face["x1"]; y1 = face["y1"]
                        gender = face["gender"]
                        emotion = face["emotion"]
                        age = face["age"]

                        face_x1 = face["face_x1"]
                        face_y1 = face["face_y1"]
                        face_x2 = face["face_x2"]
                        face_y2 = face["face_y2"]

                        text_person = f"P:{face['id_people']}; C:{face['conf']:.2f}; G:{gender}; A:{age}; E:{emotion};"
                        
                        # Draw bounding box face
                        cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
                        # Put text details in face of people
                        cv2.putText(frame, text_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                output_video.write(frame)

            else:
                cap.release()
                output_video.release()
                break
        
        total_time = time.time() - start_time
        output_video_path = "./static/output/output.mp4"
        output_video_to_browser = "./static/output/output_browser.mp4"

        try:
            cmd = f"ffmpeg -y -i {output_video_path} -vcodec libx264 -crf 23 -preset fast -pix_fmt yuv420p {output_video_to_browser}"
            os.system(cmd)
        except:
            print("FFmpeg failed to convert video")

            
        return {
                "processed_file": "./static/output/output_browser.mp4",
                "download_file": "./static/output/output.mp4",
                "stats": {
                    "people_count": len(tracked_people),
                    "persons" : persons_stats,
                    "time_run_inference": total_time
                },
                "type": "video"
            }

    def process_image(self, file_path, options, persons_stats, tracked_people):
        #Run image
        image = cv2.imread(file_path)
        #Run YOLO
        start_time = time.time()
        results_yolo = self.yolo_model.track(source=image, classes=[0] , tracker="bytetrack.yaml", conf=0.5)
                            
        for result in results_yolo:
            boxes = result.boxes
            for detection in boxes:

                # Get area of person
                x1, y1, x2, y2 = map(int, detection.xyxy.cpu().numpy()[0])
                #Extract confidence
                conf = detection.conf.cpu().item()

                #Track id
                track_id = int(detection.id.cpu().item()) if detection.id is not None else -1
                
                # Get the max number of people in the video
                tracked_people.add(track_id)
                parts_text = [f"P:{track_id}", f"C:{conf:.2f}"]
                if options["gender"] or options["age"] or any(options["emotions"].values()):

                    # Crop the person image for emotion analyze
                    person = image[y1:y2, x1:x2]
                    face = self.app_insightface.get(person)

                    # Bounding box of face
                    fx1, fy1, fx2, fy2 = map(int, face[0].bbox)

                    if options["gender"]:
                        gender = "male" if face[0].gender == 1 else "female"
                        parts_text.append(f"G:{gender}")
                    else:
                        gender = None
                    
                    if options["age"]:
                        age = face[0].age
                        parts_text.append(f"G:{age}")
                    else:
                        age = None

                    #Analyze only the face
                    if any(options["emotions"].values()):
                        emotion = self.analyze_emotion(person[fy1:fy2, fx1:fx2])
                    else:
                        emotion = None

                    if options["emotions"][emotion]:
                        parts_text.append(f"G:{emotion}")

                    # Face relative to entire image
                    face_x1 = x1 + fx1
                    face_y1 = y1 + fy1
                    face_x2 = x1 + fx2
                    face_y2 = y1 + fy2
                    
                    #<---- AI Help ---->
                    text_person = "; ".join(parts_text) + ";"
                    #<----------------->

                    # Draw bounding box face
                    cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)

                    # Draw bounding box person
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text for each person
                    cv2.putText(image, text_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    #<--- AI Help to make possible register a list of emotions into a dict of dicts
                    if track_id not in persons_stats:
                        persons_stats[track_id] = {
                            "confidence": conf, "gender": gender, "age": age, "emotions": []
                        }
                        if options["emotions"][emotion]:
                            persons_stats[track_id]["emotions"].append(emotion)
                    else:
                        if options["emotions"][emotion]:
                            persons_stats[track_id]["emotions"].append(emotion)
        path_image = os.path.join(BASE_DIR, "static", "output", "image.jpg")
        cv2.imwrite(path_image, image)
        total_time = time.time() - start_time

        return {
            "processed_file": "/static/output/image.jpg",
            "stats": {
                "people_count": len(tracked_people),
                "persons" : persons_stats,
                "time_run_inference": total_time
            },
            "type": "image"
        }