import threading
import cv2
import numpy as np

from Detectors.IDetector import IDetector
from time import time

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter

skip_frames = 3


class VideoDetector(IDetector):
    """
    Detects objects on videos.
    """
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        super().__init__(dataSource, model, classes)

    def create_video_writer(self, player, outFile: str):
        xShape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        yShape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourCC = cv2.VideoWriter_fourcc(*"MJPG")

        return cv2.VideoWriter("detections\\" + outFile, fourCC, 20, (xShape, yShape))

    def wait_for_key_press(self):
        input("Press any key to stop the detection...")

    def detect(self, source: str, outFile: str):
        keyThread = threading.Thread(target=self.wait_for_key_press)
        keyThread.start()

        player = self.dataSource.load_data(source)
        out = self.create_video_writer(player=player, outFile=outFile)

        fps = 0
        frame_counter = 0
        plotter = Plotter(self.classes)
        while keyThread.is_alive():
            startTime = time()
            ret, frame = player.read()

            if frame_counter % skip_frames == 0:
                data = super().score_frame(frame=frame)

            frame_counter += 1

            if not ret:
                break

            endTime = time()
            fps = 1/np.round(endTime - startTime, 3)

            frame = plotter.plot(frame=frame, fps=fps,
                                 labels=data["labels"], cords=data["coords"], confidence=data["confidence"])

            out.write(frame)

        out.release()
        player.release()
