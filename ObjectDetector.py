import cv2
import torch
import numpy as np

from ArgParser.ArgParser import parse
from DataLoaders.YTLoader import YTLoader
from DataLoaders.ImageLoader import ImageLoader
from DataLoaders.VideoLoader import VideoLoader
from MediaDetector.MediaDetector import getMediaType
from Detectors.IDetector import IDetector
from Detectors.VideoDetector import VideoDetector
from Detectors.ImageDetector import ImageDetector
from Plotters.Plotter import Plotter


class ObjectDetector:
    """
    Uses Yolov5 object detection algorithm to detect certain objects and highlights them through OpenCV.
    """

    def __init__(self, path: str, outFile: str):
        """
        Parameters
        ----------
        path:
            The path from which the object detection file is loaded.
            It can be a yt video, image ...
        outFile:
            The path to the file to which the video detection result will be saved.
        """
        self.path = path
        self.outFile = outFile

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flags = parse()

        self.model = self.loadModel()
        self.classes = self.model.names

        self.setDataLoader()

        print(f"Classes: {self.classes}")

        print(f"{self.device} is used for detection.\n")

    def setDataLoader(self):
        """
        Checks the file type from the path variable and sets the class data loader.
        Data can be loaded from YT videos, images, .mp4 videos ...
        """
        mediaType = getMediaType(self.path)

        if mediaType == "link":
            self.dataLoader = YTLoader()
            self.detector = VideoDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), classes=self.classes)

        elif mediaType == "video":
            self.dataLoader = VideoLoader()
            self.detector = VideoDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), classes=self.classes)

        elif mediaType == "image":
            self.dataLoader = ImageLoader()
            self.detector = ImageDetector(dataSource=self.dataLoader, model=self.model.to(
                self.device), classes=self.classes)

        else:
            print("File type not supported!")
            return -1

    def loadDetectionFile(self, path: str):
        """
        Loads the data for detection in the correct format.
        """
        return self.dataLoader.loadData(path)

    def loadModel(self):
        model = torch.hub.load("ultralytics/yolov5",
                               "yolov5s", pretrained=True)

        if len(self.flags["classes"]) > 0:
            model.classes = self.flags["classes"]

        if self.flags["conf"] is not None:
            model.conf = self.flags["conf"]

        return model

    def detect(self):
        self.detector.detect(source=self.path, outFile=self.outFile)

# TODO: fix displayDetectionVideo
    #     self.displayDetectionVideo()

    # def displayDetectionVideo(self):
    #     sleep(2)
    #     print("In display")
    #     cap = cv2.VideoCapture(self.outFile)
    #     print(cap.isOpened())

    #     if not cap.isOpened():
    #         print(f"Error reading {self.outFile}!")
    #         return -1

    #     # Read until video is completed
    #     while(cap.isOpened()):

    #     # Capture frame-by-frame
    #         ret, frame = cap.read()
    #         print(ret, frame)
    #         if ret == True:
    #         # Display the resulting frame
    #             cv2.imshow('Frame', frame)

    #         # Press Q on keyboard to exit
    #             if keyboard.read_key() == "q":
    #                 break

    #     # Break the loop
    #         else:
    #             break

    #     # When everything done, release
    #     # the video capture object
    #     cap.release()


if __name__ == "__main__":
    detector = ObjectDetector(
        "https://www.youtube.com/watch?v=NyLF8nHIquM", "video1.avi")
    detector.detect()
