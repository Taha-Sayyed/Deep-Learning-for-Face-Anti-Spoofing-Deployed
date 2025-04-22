import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import time
from PIL import Image
import tempfile
from pathlib import Path
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Disable GPU and configure TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Try-except block for OpenCV's CUDA settings
try:
    cv2.setUseOptimized(True)
    # Omit CUDA settings for OpenCV since we're using CPU
except:
    pass

# Import our CPU-optimized model utility functions
from model_utils_cpu import FaceAntiSpoofModel, detect_face

# Set page config
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon="🔒",
    layout="centered"
)

# Initialize session state variables
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'face_model' not in st.session_state:
    st.session_state.face_model = None

def apply_label_to_frame(frame, label, confidence):
    """Apply prediction label to the video frame"""
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Set color based on prediction
    if label == "REAL":
        color = (0, 255, 0)  # Green for real
    else:
        color = (0, 0, 255)  # Red for fake
    
    # Create the text to display
    text = f"{label}: {confidence:.2f}"
    
    # Calculate text position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30
    
    # Draw background rectangle
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

class FaceAntiSpoofProcessor(VideoProcessorBase):
    def __init__(self, face_model):
        self.face_model = face_model
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect face in the frame
        frame_with_rect, face_roi = detect_face(img)
        
        # If face detected, predict whether it's real or fake
        if face_roi is not None:
            # Predict
            with tf.device('/CPU:0'):
                label, confidence = self.face_model.predict(face_roi)
            
            # Apply label to frame
            frame_with_label = apply_label_to_frame(frame_with_rect, label, confidence)
            
            # Return the processed frame
            return av.VideoFrame.from_ndarray(frame_with_label, format="bgr24")
        else:
            # If no face detected, just display the frame with a message
            text = "No face detected"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Return the processed frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Set up page title and description
    st.title("Face Anti-Spoofing Detection")
    st.markdown("This application detects whether a face is real or a spoofing attack.")
    
    # Model path (ensure it's relative to streamlit app's root)
    model_path = "model_checkpoints/model.08-1.00.keras"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at path: {model_path}")
        st.info("Please make sure the model file is available in the specified directory.")
        return
    
    # Load the model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            # Force TensorFlow to use CPU
            with tf.device('/CPU:0'):
                st.session_state.face_model = FaceAntiSpoofModel(model_path)
                
                if st.session_state.face_model.model is None:
                    st.error("Failed to load model. Please check model path and try again.")
                    return
                
                st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
    
    # Choose input method
    input_method = st.radio("Choose input method:", ["Webcam", "Upload Image"])
    
    if input_method == "Webcam":
        # Configure WebRTC
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create WebRTC streamer
        st.write("Click the 'START' button below to access your webcam")
        webrtc_ctx = webrtc_streamer(
            key="face-anti-spoofing",
            video_processor_factory=lambda: FaceAntiSpoofProcessor(st.session_state.face_model),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Display instructions
        if webrtc_ctx.state.playing:
            st.info("The webcam is now active. The model will analyze your face in real-time.")
        else:
            st.warning("Click 'START' to begin face anti-spoofing detection with your webcam.")
    
    elif input_method == "Upload Image":
        # Allow user to upload an image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            with st.spinner("Processing image..."):
                # Create a temporary file to save the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Read the image with OpenCV
                image = cv2.imread(temp_path)
                
                # Convert from BGR to RGB for display
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display the uploaded image
                st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
                
                # Detect face in the image
                image_with_rect, face_roi = detect_face(image)
                
                # If face detected, predict whether it's real or fake
                if face_roi is not None:
                    # Predict
                    with tf.device('/CPU:0'):
                        label, confidence = st.session_state.face_model.predict(face_roi)
                    
                    # Apply label to image
                    result_image = apply_label_to_frame(image_with_rect, label, confidence)
                    
                    # Convert from BGR to RGB for display
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Display the result
                    st.image(result_rgb, caption=f"Result: {label} (Confidence: {confidence:.2f})", use_column_width=True)
                else:
                    st.warning("No face detected in the uploaded image.")
                
                # Remove the temporary file
                os.unlink(temp_path)

    # Add information about the app
    st.markdown("---")
    st.markdown("""
    ## About this app
    This app uses a deep learning model to detect face spoofing attacks. 
    
    It can distinguish between:
    - Real faces (live person)
    - Fake faces (photos, videos, masks, etc.)
    
    ### Technical Details
    - The model is running on CPU for Streamlit Cloud compatibility
    - Webcam access is enabled through WebRTC
    - You can also upload images for analysis
    """)

if __name__ == "__main__":
    main() 