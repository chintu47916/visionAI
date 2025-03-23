import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import pickle
import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import tempfile
import torch
import torchvision.transforms as T
from PIL import Image

# Initialize pygame mixer for voice output
pygame.mixer.init()
recognizer = sr.Recognizer()

st.title("👀 AI Human Eye - Camera Vision with Object Detection and Navigation")
run = st.checkbox("Start Camera")

# Load known face encodings
KNOWN_FACES_FILE = "known_faces.pkl"
if os.path.exists(KNOWN_FACES_FILE):
    try:
        with open(KNOWN_FACES_FILE, "rb") as f:
            known_faces = pickle.load(f)
        if not isinstance(known_faces, dict) or "names" not in known_faces or "encodings" not in known_faces:
            raise ValueError("Invalid format")
    except (EOFError, ValueError, KeyError):
        known_faces = {"names": [], "encodings": []}  # Reset if file is corrupted
else:
    known_faces = {"names": [], "encodings": []}  # Store names & encodings separately

# Load YOLO model for object detection
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to save known faces
def save_known_faces():
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(known_faces, f)

# Function to play audio
def speak(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(text)
        audio_file = temp_audio.name
        tts.save(audio_file)
    
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait until audio finishes playing
        time.sleep(0.1)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    os.remove(audio_file)  # Delete file after playback

# Function to capture voice input
def get_voice_name():
    with sr.Microphone() as source:
        st.write("Please say your name...")
        try:
            audio = recognizer.listen(source, timeout=5)
            name = recognizer.recognize_google(audio).strip()
            return name
        except sr.UnknownValueError:
            st.write("Could not understand audio. Try again.")
        except sr.RequestError:
            st.write("Speech recognition service unavailable.")
    return None

# Open Camera
camera_index = 0
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    camera_index = 1  # Try an alternative camera index
    camera = cv2.VideoCapture(camera_index)

if not camera.isOpened():
    st.error("Camera not found! Please check your device.")
else:
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            name = "Unknown"
            if known_faces["encodings"]:
                distances = face_recognition.face_distance(known_faces["encodings"], face_encoding)
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.5:  # Use threshold to avoid misidentifications
                    name = known_faces["names"][best_match_index]
                    speak(f"{name} present")
                else:
                    name = get_voice_name()
                    if name:
                        known_faces["names"].append(name)
                        known_faces["encodings"].append(face_encoding)
                        save_known_faces()
                        speak(f"Hello {name}")
            else:
                name = get_voice_name()
                if name:
                    known_faces["names"].append(name)
                    known_faces["encodings"].append(face_encoding)
                    save_known_faces()
                    speak(f"Hello {name}")
        
        # Object detection
        img_np = np.array(rgb_frame)
        results = detection_model(img_np)
        
        # Draw bounding boxes
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{detection_model.names[int(cls)]} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Identify potholes, manholes, and steps for navigation
            if "pothole" in label or "manhole" in label:
                speak("Caution! Pothole detected ahead. Step aside carefully.")
            elif "stairs" in label:
                speak("Stairs detected. Hold the railing and step down carefully.")
            
            # Identify money and track amount
            if "money" in label:
                speak("Money detected. Scanning amount...")
                # Money counting logic can be added here
            
            # Announce detected objects
            speak(f"Detected {label}")
        
        # Display on Streamlit
        st.image(frame, channels="BGR")

    camera.release()
