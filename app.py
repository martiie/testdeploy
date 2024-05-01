import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from utils import FPSmetric
from engine import Engine
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

st.title("My first Streamlit app")
st.write("Hello, Phithak")
facenet = FaceNet(
        detector = MPFaceDetection(),
        onnx_model_path = "models/faceNet.onnx", 
        anchors = "faces",
        force_cpu = True,
        threshold=0.15,
    )
engine = Engine(videocap=0, show=True, custom_objects=[facenet, FPSmetric()])

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = engine.custom_processing(engine.flip(img))
    return av.VideoFrame.from_ndarray(img, format="bgr24")

#webrtc_streamer(key="example")#, video_frame_callback=callback)
webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

st.write("Hello, TOP")
