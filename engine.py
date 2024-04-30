import cv2
import stow
import typing
import numpy as np
from tqdm import tqdm 
from faceNet.faceNet import FaceNet
from selfieSegmentation import MPSegmentation
from faceDetection import MPFaceDetection

class Engine:
    """Object to process webcam stream, video source or images
    All the processing can be customized and enchanced with custom_objects
    """
    def __init__(
        self, 
        image_path: str = "",
        video_path: str = "", 
        webcam_id: int = 0,
        videocap: int =0,
        show: bool = False,
        flip_view: bool = False,
        custom_objects: typing.Iterable = [],
        output_extension: str = 'out',
        start_video_frame: int = 0,
        end_video_frame: int = 0,
        break_on_end: bool = False,
        ) -> None:
        self.videocap = videocap
        self.video_path = video_path
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects
        self.output_extension = output_extension
        self.start_video_frame = start_video_frame
        self.end_video_frame = end_video_frame
        self.break_on_end = break_on_end

    def flip(self, frame: np.ndarray) -> np.ndarray:
        if self.flip_view:
            return cv2.flip(frame, 1)

        return frame

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:
        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def display(self, frame: np.ndarray, webcam: bool = False, waitTime: int = 1) -> bool:
        if self.show:
            cv2.imshow('TOP', frame)
            k = cv2.waitKey(waitTime)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False
                 
            if webcam:
                if k & 0xFF == ord('a'):
                    for custom_object in self.custom_objects:
                        # change background to next with keyboar 'a' button
                        if isinstance(custom_object, MPSegmentation):
                            custom_object.change_image(True)
                elif k & 0xFF == ord('d'):
                    for custom_object in self.custom_objects:
                        # change background to previous with keyboar 'd' button
                        if isinstance(custom_object, MPSegmentation):
                            custom_object.change_image(False)

        return True

    def process_image(
        self, 
        image: typing.Union[str, np.ndarray] = None, 
        output_path: str = None
        ) -> np.ndarray:
        if image is not None and isinstance(image, str):
            if not stow.exists(image):
                raise Exception(f"Given image path doesn't exist {self.image_path}")
            else:
                extension = stow.extension(image)
                if output_path is None:
                    output_path = image.replace(f".{extension}", f"_{self.output_extension}.{extension}")
                image = cv2.imread(image)

        image = self.custom_processing(self.flip(image))

        cv2.imwrite(output_path, image)

        self.display(image, waitTime=0)

        return image

    def process_webcam(self, return_frame: bool = False) -> typing.Union[None, np.ndarray]:
        # Create a VideoCapture object for given webcam_id
        cap = cv2.VideoCapture(self.webcam_id)
        while cap.isOpened():  
            success, frame = cap.read()
            if not success or frame is None:
                print("Ignoring empty camera frame.")
                continue

            if return_frame:
                break

            frame = self.custom_processing(self.flip(frame))

            if not self.display(frame, webcam=True):
                break

        else:
            raise Exception(f"Webcam with ID ({self.webcam_id}) can't be opened")

        cap.release()
        return frame

    def check_video_frames_range(self, fnum):
        if self.start_video_frame and fnum < self.start_video_frame:
            return True

        if self.end_video_frame and fnum > self.end_video_frame:
            return True
        
        return False

    def process_video(self) -> None:
        if not stow.exists(self.video_path):
            raise Exception(f"Given video path doesn't exists {self.video_path}")

        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            raise Exception(f"Error opening video stream or file {self.video_path}")

        # Capture video details
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer in the same location as original video
        output_path = self.video_path.replace(f".{stow.extension(self.video_path)}", f"_{self.output_extension}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        # Read all frames from video
        for fnum in tqdm(range(frames)):
            # Capture frame-by-frame
            success, frame = cap.read()
            if not success:
                break

            if self.check_video_frames_range(fnum):
                out.write(frame)
                if self.break_on_end and fnum >= self.end_video_frame:
                    break
                continue

            frame = self.custom_processing(self.flip(frame))

            out.write(frame)

            if not self.display(frame):
                break

        cap.release()
        out.release()

    def process_videocap(self,frame, return_frame: bool = False) -> typing.Union[None, np.ndarray]:
            frame = self.custom_processing(self.flip(frame))
            return frame
    
    def run(self):
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image(self.image_path)
        elif self.videocap:
            self.process_videocap(self.videocap)    
        else:
            self.process_webcam()