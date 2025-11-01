import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np
import cv2
import streamlit as st


@st.cache_resource
def load_faster_rcnn(model_path: str, num_classes: int = 2):
    """
    Load Faster R-CNN model with pretrained weights.
    Cached using Streamlit for performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights from URL (Hugging Face)
    state_dict = torch.hub.load_state_dict_from_url(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def detect_faster_rcnn(model, frame, conf=0.5):
    """
    Run inference using Faster R-CNN on an image or video frame.
    Returns a visualized image (numpy array) with thicker boxes and confidence labels.
    """
    device = next(model.parameters()).device

    # Convert to PIL image if needed
    if isinstance(frame, np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
    elif isinstance(frame, Image.Image):
        img_pil = frame.convert("RGB")
    else:
        raise ValueError("Unsupported input type for 'frame'.")

    # Preprocess
    img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()

    # Filter by confidence
    keep = scores >= conf
    boxes = boxes[keep]
    scores = scores[keep]

    # Skip drawing if no detections
    if len(boxes) == 0:
        return np.array(img_pil)

    # Create label list with "person" + confidence score
    labels = [f"person {s:.2f}" for s in scores]

    # Draw bounding boxes — thicker line (width=4)
    img_tensor = (F.to_tensor(img_pil) * 255).byte()
    result_tensor = draw_bounding_boxes(
        img_tensor,
        boxes,
        labels=labels,
        colors="red",
        width=1,
        font_size=14
    )

    # Convert to numpy for Streamlit display
    result_img = F.to_pil_image(result_tensor)
    return np.array(result_img)

