import os
from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def load_yolo(model_path):
    return YOLO(model_path)

def detect_yolo(model, frame, conf=0.5):
    results = model(frame, conf=conf)
    return results[0].plot()