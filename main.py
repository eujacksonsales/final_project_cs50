import shutil
import os
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import your custom CV logic
from cv_engine.processor import CVProcessor

app = FastAPI()

# 1. MOUNT STATIC FILES
# This allows HTML to access CSS, JS, and the processed images in the 'output' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. SETUP TEMPLATES
templates = Jinja2Templates(directory="templates")

# 3. INITIALIZE CV ENGINE
# We do this globally so models load only once!
cv_brain = CVProcessor()

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main page (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_media(
    request: Request,
    file: UploadFile = File(...),
    # Checkbox inputs from your wireframe (Form parameters)
    count_people: bool = Form(False),
    count_gender: bool = Form(False),
    count_age: bool = Form(False),
    emotion_joy: bool = Form(False),
    emotion_anger: bool = Form(False),
    # Add the rest of your emotions here based on your wireframe...
):
    """
    Handles the file upload and triggers the CV processing.
    """
    
    # TODO Step 1: Save the uploaded file to the 'uploads' folder
    # I've written the safe file handling for you here to prevent errors.
    upload_dir = "uploads"
    file_location = f"{upload_dir}/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # TODO Step 2: Pack the preferences into a dictionary
    options = {
        "people": count_people,
        "gender": count_gender,
        "age": count_age,
        "emotions": {
            "joy": emotion_joy,
            "anger": emotion_anger,
            # ...
        }
    }

    # TODO Step 3: Call your CV muscle
    # This calls the function you created in cv_engine/processor.py
    results = cv_brain.process_media(file_location, options)

    # TODO Step 4: Return the result to the frontend
    # We re-render the index.html, but this time we pass the 'results' variable.
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": results, 
        "original_file": file.filename
    })