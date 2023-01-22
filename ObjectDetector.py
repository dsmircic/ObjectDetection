import torch
import numpy as np
import os
import cv2

from time import sleep
from ArgParser.ArgParser import parse
from DataLoaders.YTLoader import YTLoader
from DataLoaders.ImageLoader import ImageLoader
from DataLoaders.VideoLoader import VideoLoader
from DataLoaders.CameraLoader import CameraLoader

from MediaDetector.MediaDetector import get_media_type

from Detectors.VideoDetector import VideoDetector
from Detectors.ImageDetector import ImageDetector


class ObjectDetector:
    """
    Uses Yolov5 object detection algorithm to detect certain objects and highlights them through OpenCV.
    """

    def __init__(self):
        """
        Parameters
        ----------
        path:
            The path from which the object detection file is loaded.
            It can be a yt video, image ...
        outFile:
            The path to the file to which the video detection result will be saved.
        """
        self.flags = parse()
        if self.flags["source"] is not None:
            self.path = self.flags["source"]
        else:
            print("Provide source file with the '--source' flag")
            return -1

        if self.flags["dest"] is not None:
            self.outFile = self.flags["dest"]
        else:
            print("Provide out file name with '--dest' flag")
            return -1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.classes = self.model.names

        self.set_data_loader()

        if not os.path.exists("detections"):
            os.makedirs("detections")

        print(f"{self.device} is used for detection.\n")

    def set_data_loader(self):
        """
        Checks the file type from the path variable and sets the class data loader.
        Data can be loaded from YT videos, images, .mp4 videos ...
        """
        mediaType = get_media_type(self.path)

        if mediaType == "link":
            self.dataLoader = YTLoader()
            self.detector = VideoDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), params=self.flags, classes=self.classes)

        elif mediaType == "video":
            self.dataLoader = VideoLoader()
            self.detector = VideoDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), params=self.flags, classes=self.classes)

        elif mediaType == "image":
            self.dataLoader = ImageLoader()
            self.detector = ImageDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), params=self.flags, classes=self.classes)

        elif mediaType == "camera":
            self.dataLoader = CameraLoader
            self.detector = VideoDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), params=self.flags, classes=self.classes)

        else:
            print("File type not supported!")
            return -1

    def load_detection_file(self, path: str):
        """
        Loads the data for detection in the correct format.
        """
        return self.dataLoader.load_data(path)

    def load_model(self):
        """
        Loads the yolov5 model from the ultralitycs/yolov5 github repo.
        """
        model = torch.hub.load("ultralytics/yolov5",
                               "yolov5s", pretrained=True)

        if len(self.flags["classes"]) > 0:
            model.classes = self.flags["classes"]

        if self.flags["conf"] is not None:
            model.conf = self.flags["conf"]

        return model

    def detect(self):
        """
        Runs the detection based on the file type (eg. link, video, image) and stores the results in the detections\\ directory.
        """
        self.detector.detect(source=self.path, outFile=self.outFile)
        self.display_detection_video()

    def display_detection_video(self):
        sleep(2)
        cap = cv2.VideoCapture("detections\\" + self.outFile)
        
        if not cap.isOpened():
            print(f"Error reading {self.outFile}!")
            return -1

        # Read until video is completed
        print("Press 'q' to stop viewing the video.")
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
            else:
                break

        # When everything done, release
        # the video capture object
        cap.release()


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.detect()
