import cv2
import torch
import numpy as np
import keyboard
import threading

from time import time
from DataLoaders.IDataLoader import IDataLoader
from DataLoaders.YTLoader import YTLoader
from DataLoaders.ImageLoader import ImageLoader
from DataLoaders.VideoLoader import VideoLoader
from MediaDetector.MediaDetector import getMediaType

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
        self.model = self.loadModel()
        self.classes = self.model.names

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
        
        elif mediaType == "video":
            self.dataLoader = VideoLoader()
        
        elif mediaType == "image":
            self.dataLoader = ImageLoader()
        
        else:
            print("File type not supported!")
            return -1

    def loadDetectionFile(self, path: str):
        """
        Loads the data for detection in the correct format.
        """
        return self.dataLoader.loadData(path)

    def loadModel(self):
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained = True)
        return model

    def scoreFrame(self, frame):
        """
        Takes a single frame as input, and scores it using the yolov5 model.

        Parameters
        ----------
        frame:
            The frame on which detection will be made.
        """

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coord

    def classToLabel(self, x):
        """
        Returns the label name in a string format from the corresponding numeric label.

        Parameters
        ----------
        x:
            The numeric label.
        """

        return self.classes[int(x)]

    def plotBoxes(self, frame):
        """
        Takes a frame and it's results as input and plots bounding boxes and labels onto the frame.

        Parameters
        ----------
        frame:
            Image frame on which the bounding boxes and labels will be plotted.
        """

        labels, cord = self.scoreFrame(frame)
        n = len(labels)
        xShape, yShape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * xShape), int(row[1] * yShape), int(row[2] * xShape), int(row[3] * yShape)
                background = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)
                cv2.putText(frame, self.classToLabel(labels[i]), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)

        return frame

    def displayFPS(self, frame, fps:float):
        colour = (255, 0, 0)
        text = "FPS: " + str(np.round(fps, 2))
        location = (20, 20)

        cv2.putText(
            frame,
            text,
            location,
            cv2.FONT_ITALIC,
            0.9,
            colour,
            2
        )

        return frame

    def createFrame(self, frame, fps):
        boxes = self.plotBoxes(frame)
        frames = self.displayFPS(boxes, fps=fps)

        return frames

    def createVideoWriter(self, player):
        xShape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        yShape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourCC = cv2.VideoWriter_fourcc(*"MJPG")

        return cv2.VideoWriter("detections\\" + self.outFile, fourCC, 20, (xShape, yShape))

    def detect(self):
        keyThread = threading.Thread(target=self.waitForKeyPress)
        keyThread.start()

        self.setDataLoader()
        player = self.dataLoader.loadData(self.path)

        out = self.createVideoWriter(player=player)

        fps = 0
        while keyThread.is_alive():
            startTime = time()
            ret, frame = player.read()
            if not ret:
                break

            frame = self.createFrame(frame, fps)
            endTime = time()
            fps = 1/np.round(endTime - startTime, 3)

            out.write(frame)

    def waitForKeyPress(self):
        input("Press any key to stop the detection...")

#TODO: fix displayDetectionVideo
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
    detector = ObjectDetector("https://www.youtube.com/watch?v=EXUQnLyc3yE", "video1.avi")
    detector.detect()