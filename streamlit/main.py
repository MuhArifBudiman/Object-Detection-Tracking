import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from yolo_utils import load_yolo, detect_yolo
from faster_rcnn_utils import load_faster_rcnn, detect_faster_rcnn
from PIL import Image


def main():
    st.title("Person Tracking")
    st.write("By: CV B (Fei Fei Li)")
    st.write("""
    - Giar
    - Fitri
    - Joe
    - Arif
    """)

    model_path = "model"

    option = st.selectbox(
        "Select Model or YOLO version",
        ["YOLOv9",
         "YOLOv8m",
         "YOLOv8n",
         "Faster R CNN"]
    )

    model_mapping = {
        "YOLOv9": "https://huggingface.co/muh-arif21/yolo_models/resolve/main/yolov9_custom.pt",
        "YOLOv8m": "https://huggingface.co/muh-arif21/yolo_models/resolve/main/yolov8m_giar.pt",
        "YOLOv8n": "https://huggingface.co/muh-arif21/yolo_models/resolve/main/yolov8n_joe.pt"
    }

    if option:
        if "yolo" in option.lower():
            if option == "YOLOv9":
                model_link = model_mapping.get("YOLOv9")
            elif option == "YOLOv8m":
                model_link = model_mapping.get("YOLOv8m")
            else:
                model_link = model_mapping.get("YOLOv8n")
            st.write("Selected model: ", option)
            model = load_yolo(model_path=model_link)
        else:
            model_link = "https://huggingface.co/muh-arif21/faster_rcnn/resolve/main/fasterrcnn_person.pth"
            st.write("Selected model: ", option)
            model = load_faster_rcnn(model_path=model_link)
    else:
        st.write("Please select your model")

        # Pilihan jenis input
    st.write("Select option")
    mode = st.radio("Choose input type:", ["Image", "Video"])

    # Jika input image
    if mode == "Image":
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Detection"):
                if "yolo" in option.lower():
                    result_img = detect_yolo(model, image)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                else:
                    result_img = detect_faster_rcnn(model, image)
                
                st.image(result_img, caption="Detection Result",
                         use_container_width=True)

    # Jika input video
    elif mode == "Video":
        uploaded_video = st.file_uploader(
            "Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            # Simpan video sementara
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            st.info("Running detection... press stop to interrupt.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if "yolo" in option.lower():
                    result_frame = detect_yolo(model, frame_rgb)
                # else:
                #     result_frame = detect_faster_rcnn(frame_rgb, model)

                stframe.image(result_frame, channels="RGB",
                              use_container_width=True)

            cap.release()
            st.success("✅ Detection finished")


if __name__ == "__main__":
    main()