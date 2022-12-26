import threading
import cv2
import numpy as np
import os

from time import time

from Detectors.IDetector import IDetector
from time import time

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter

buffer_size = 1


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
        cFrame = 0
        while keyThread.is_alive():
            start_time = time()
            ret, frame = player.read()

            labels, cord = super().score_frame(frame=frame)
            plotter = Plotter(self.classes)

            if not ret:
                break

            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)

            frame = plotter.plot(frame=frame, fps=fps,
                                 labels=labels, cords=cord)

            out.write(frame)
            self.write_frame_to_buffer(frame=frame, current_frame=cFrame)
            cFrame += 1

        out.release()
        player.release()

    def write_frame_to_buffer(self, frame, current_frame):
        # cv2.imwrite("buffer\\buffer" + str(int(current_frame %
        #             buffer_size)) + ".jpg", frame)
        cv2.imwrite("buffer\\buffer.jpg", frame)
