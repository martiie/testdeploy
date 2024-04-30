import cv2
import stow
import typing
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime
import os
import csv
from record import Record

class FaceNet:
    def __init__(
        self, 
        detector: object,
        onnx_model_path: str = "models/faceNet.onnx", 
        anchors: typing.Union[str, dict] = 'faces',
        force_cpu: bool = False,
        threshold: float = 0.5,
        color: tuple = (255, 255, 255),
        thickness: int = 2,
        ) -> None:
        if not stow.exists(onnx_model_path):
            raise Exception(f"Model doesn't exists in {onnx_model_path}")

        self.detector = detector
        self.threshold = threshold
        self.color = color
        self.thickness = thickness

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        providers = providers if ort.get_device() == "GPU" and not force_cpu else providers[::-1]

        self.ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

        self.input_shape = self.ort_sess._inputs_meta[0].shape[1:3]
        
        self.anchors = self.load_anchors(anchors) if isinstance(anchors, str) else anchors

    def normalize(self, img: np.ndarray) -> np.ndarray:
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def l2_normalize(self, x: np.ndarray, axis: int = -1, epsilon: float = 1e-10) -> np.ndarray:
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def detect_save_faces(self, image: np.ndarray, output_dir: str = "faces",names='new'):
        face_crops = [image[t:b, l:r] for t, l, b, r in self.detector(image, return_tlbr=True)]

        if face_crops == []: 
            return False

        stow.mkdir(output_dir)

        for index, crop in enumerate(face_crops):
            output_path = stow.join(output_dir, f"{names}.png")#"face_{str(index)}.png"
            cv2.imwrite(output_path, crop)
            print("Crop saved to:", output_path)

        self.anchors = self.load_anchors(output_dir)
        
        return True

    def load_anchors(self, faces_path: str):
        anchors = {}
        if not stow.exists(faces_path):
            return {}

        for face_path in stow.ls(faces_path):
            anchors[stow.basename(face_path)] = self.encode(cv2.imread(face_path.path))

        return anchors

    def encode(self, face_image: np.ndarray) -> np.ndarray:
        face = self.normalize(face_image)
        face = cv2.resize(face, self.input_shape).astype(np.float32)

        encode = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(face, axis=0)})[0][0]
        normalized_encode = self.l2_normalize(encode)

        return normalized_encode

    def cosine_distance(self, a: np.ndarray, b: typing.Union[np.ndarray, list]) -> np.ndarray:
        if isinstance(a, list):
            a = np.array(a)

        if isinstance(b, list):
            b = np.array(b)

        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    def draw(self, image: np.ndarray, face_crops: dict):
        for value in face_crops.values():
            t, l, b, r = value["tlbr"]
            cv2.rectangle(image, (l, t), (r, b), self.color, self.thickness)
            cv2.putText(image, stow.name(value['name']), (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, self.thickness)
            cv2.putText(image, stow.name(value['change']), (l+200, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, self.thickness)
        Record.record_to_csv(1,face_crops)
        return image

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        face_crops = {index: {"name": "Unknown", "tlbr": tlbr,"change":0} for index, tlbr in enumerate(self.detector(frame, return_tlbr=True))}
        for key, value in face_crops.items():
            t, l, b, r = value["tlbr"]
            face_encoding = self.encode(frame[t:b, l:r])
            distances = self.cosine_distance(face_encoding, list(self.anchors.values()))
            face_crops[key]["change"] = str(100-distances[0]*100)
            if np.max(distances) > self.threshold:
                face_crops[key]["name"] = list(self.anchors.keys())[np.argmax(distances)]

        frame = self.draw(frame, face_crops)

        return frame