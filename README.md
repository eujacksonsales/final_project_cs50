# DeepMetrics: Advanced Demographic & Emotion Analytics

#### Video Demo:

## Project Overview

**DeepMetrics** is a high-performance web-based computer vision application designed to analyze images and video streams in a post-processing environment. The system utilizes a hybrid pipeline of deep learning models to detect people within media and extract granular demographic data, including **Gender**, **Age**, and **Emotional State**.

As a Computer Vision Engineer, my goal for this project was not simply to use a pre-made API, but to architect a robust, modular inference engine capable of running heavy neural networks on a local server. The project bridges the gap between modern web development (using **FastAPI**) and high-performance AI inference (using **YOLOv8n**, **InsightFace**, and **OpenVINO**).

The application solves a common problem in the analytics industry: getting detailed demographic data from video footage without relying on expensive cloud APIs. Whether for retail analytics, crowd monitoring, or user experience research, DeepMetrics provides a local, privacy-centric solution for extracting metadata from visual inputs.

## Technical Architecture & Stack

The project is built on a monolithic architecture where the backend serves both the API endpoints and the frontend templates. This decision was made to ensure low latency between the file upload process and the inference engine.

### Core Technologies:

  * **Python 3.10+**: The backbone of the application.
  * **FastAPI**: Chosen over Flask for its modern asynchronous capabilities, automatic data validation (via Pydantic), and superior performance benchmarks.
  * **OpenCV (cv2)**: Used for all image manipulation, video frame extraction, and drawing bounding boxes/annotations.
  * **Jinja2**: Used for server-side rendering of the HTML and CSS frontend.

### The Computer Vision Pipeline (The "Brain"):

This is the core complexity of the project. I utilized a multi-stage inference pipeline:

1.  **Object Detection (YOLOv8):** I employed the `ultralytics` YOLOv8 Nano model to detect "Person" classes. This model was selected for its balance between speed and accuracy. It tracks subjects across video frames using the ByteTrack algorithm.
2.  **Facial Attribute Analysis (InsightFace):** Once a person is detected, the system crops the region of interest (ROI) and passes it to `InsightFace` (using the `buffalo_l` model). This lightweight ResNet model estimates Gender and Age.
3.  **Emotion Recognition (OpenVINO):** For emotion detection, I integrated Intel's OpenVINO toolkit. This allows the system to run specific emotion recognition models optimized for CPU/GPU inference, classifying faces into categories like Happy, Neutral, Sad, Anger, and Surprise.

## Project Structure

The project follows a modular "Separation of Concerns" design pattern to ensure scalability and clean code.
<!-- AI Help-->
```text
/project_root
│
├── main.py                # The Application Entry Point
├── requirements.txt       # Dependency management
├── README.md              # Description of all project
├── /cv_engine             # The Computer Vision Logic Module
│   ├── __init__.py
│   ├── processor.py       # Core inference class (CVProcessor)
│   └── /models            # Local storage for OpenVINO/YOLO weights
│       └── /emotion
|           ├── /FP16      # Model with Half Precision
|           ├── /FP16-INT8 # Model for edge devices like Raspberry Pi 5
|           └── /FP32      # Model with Full Precison(best accuracy)
├── /static                # Static assets
│   ├── styles.css         # Bootstrap overrides and custom styling
│   └── /output            # Storage for processed images/videos
│
├── /templates             # Jinja2 HTML Templates
│   └── index.html         # The main user interface
│
└── /uploads               # Temporary storage for raw user inputs
```
<!-- -->

### Detailed File Descriptions

  * **`main.py`**: This file initializes the FastAPI app. It handles the `GET /` route to render the UI and the `POST /analyze` route to handle file uploads. Crucially, it manages the dependency injection of the `CVProcessor` class, ensuring that heavy AI models are loaded only once during server startup, rather than being reloaded for every user request.

  * **`cv_engine/processor.py`**: This is the engine room of the project. It contains the `CVProcessor` class. This class handles:

      * Loading models to the GPU (CUDA) or CPU.
      * Differentiating between Image and Video processing logic.
      * Managing the mathematical logic for cropping faces and normalizing coordinates.
      * Drawing the visual annotations (bounding boxes and text) on the frames.
      * Generating the statistical summary returned to the frontend.

  * **`templates/index.html`**: A responsive frontend built with **Bootstrap 5**. It uses a unified form for uploading media and selecting analysis preferences (e.g., "Check for Gender" or "Check for Emotion"). It features a split-screen results view (Input vs. Output) and a dynamic list of statistics generated by the Jinja2 template engine.

## Design Decisions & Optimization Challenges

The most significant challenge during this project was **performance optimization**.

### 1\. The "180 Seconds" Bottleneck

In the initial prototype, processing a 15-second video took nearly 3 minutes (180 seconds). This was unacceptable for a user experience. The lag was caused by running complex facial analysis on every single frame of a 1080p video.

### 2\. The Solution: Smart Inference Caching

To solve this, I implemented several engineering optimizations:

  * **Frame Skipping:** The YOLO detector tracks people after a interval of frames, the same logic is to analyze the face, after a interval, because this reduce the number of times that the models make inferences.
  * **Resolution Scaling:** I implemented a resizing protocol that scales video frames down to 640px for inference while maintaining aspect ratio. This allowed the models to run significantly faster without losing detection accuracy.

**Result:** These changes reduced the processing time from **180 seconds down to \~30 seconds** for the same video file, achieving near-real-time performance.

### 3\. GPU Acceleration (CUDA vs OpenVINO)

A major hurdle was configuring the environment. `InsightFace` requires `onnxruntime-gpu` to access NVIDIA CUDA cores. I encountered conflicts where the standard `onnxruntime` CPU package was overriding the GPU version. I had to carefully curate the `requirements.txt` and implement runtime checks in `processor.py` to verify that `CUDAExecutionProvider` was active.

## What I Learned

This project was a significant step up from standard academic exercises. While I came in with foundational knowledge of **OpenCV** and basic Computer Vision concepts, building a production-ready inference pipeline required mastering several advanced libraries and architectural patterns:

* **Advanced Model Integration (YOLOv8 & Tracking):**
    I moved beyond simple object detection to implementing **Object Tracking**. I learned how to use YOLOv8's `track()` method with the **ByteTrack** algorithm to maintain persistent IDs for people across video frames, which was essential for creating a coherent user output.

* **High-Performance Face Analysis (InsightFace):**
    I learned the importance of model selection for specific tasks. Transitioning from generic wrappers to **InsightFace** allowed me to work directly with optimized ArcFace/RetinaFace models. I gained a deeper understanding of how input resolution (`det_size`).

* **Hardware Acceleration (OpenVINO & ONNX):**
    This was the steepest learning curve. I learned how to manage **Execution Providers** in ONNX Runtime to force calculations onto the GPU (CUDA). I also worked with Intel's **OpenVINO** Intermediate Representation (IR) formats (`.xml` and `.bin` files) to run optimized emotion detection models, understanding the difference between standard model weights and compiled runtime cores.

* **Robust File Handling (Python Magic):**
    I learned that relying on file extensions (like `.jpg` or `.mp4`) is insecure and unreliable. I integrated **`python-magic`** to inspect file headers and validate MIME types programmatically, ensuring the backend handles binary data correctly regardless of how the file is named.

* **Asynchronous Web Architecture:**
    Coming from a synchronous background, using **FastAPI** taught me how to handle non-blocking operations. I learned to structure a project where heavy blocking code (CV inference) coexists with a responsive web server.

## Installation & Usage

To run DeepMetrics locally, you need a Python environment and (optionally) an NVIDIA GPU for best performance.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/eujacksonsales/final_project_cs50.git
    cd final_project_cs50
    ```
<!-- AI Help -->
2.  **Install Dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Download Models:**
    The system will automatically attempt to download the necessary YOLO and InsightFace models on the first run. Ensure you have an internet connection.

4.  **Run the Server:**

    ```bash
    uvicorn main:app --reload
    ```
<!-- -->
5.  **Access the Application:**
    Open your browser and navigate to `http://127.0.0.1:8000`.

## Future Improvements

While the current version is a fully functional MVP, there are several areas for future expansion:

  * **Live Webcam Streaming:** Currently, the system uses a "Record & Upload" workflow. Implementing WebSockets would allow for true real-time streaming analysis.
  * **Database Integration:** Storing the statistical results (e.g., "50% of visitors today were happy") in a SQL database (like SQLite or PostgreSQL) to create a historical dashboard.
  * **Dockerization:** Packaging the application and its complex CUDA dependencies into a Docker container for easier deployment.

### AI Assistance Disclaimer

In accordance with the CS50 Academic Honesty policy, I utilized AI-based software (specifically LLMs) as a productivity amplifier during the development of DeepFace.

* **Role of AI:** I operated as the primary architect and lead engineer. I used AI tools (acting as a "Senior Architect/Mentor") to help brainstorm the directory structure, debug complex CUDA configuration errors, and scaffold the initial boilerplate code for FastAPI and Jinja2 templates.
* **Verification:** All logic, particularly the Computer Vision inference loops (`processor.py`) and the optimization strategies (frame skipping, caching), was reviewed, tested, and refined by me to ensure accuracy and performance. The core implementation details and engineering decisions remain my own work.

## Credits

  * **CS50 Team:** For the foundational knowledge in computer science, web development and algorithmic thinking.
  * **Ultralytics:** For the YOLOv8 implementation.
  * **DeepInsight:** For the InsightFace library.
  * **OpenVINO:** For the optimized emotion recognition models.
