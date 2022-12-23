import threading
import cv2
import numpy as np

from time import time

from Detectors.IDetector import IDetector
from time import time

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter

bufferSize = 7

class VideoDetector(IDetector):
    """
    Detects objects on videos.
    """
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        super().__init__(dataSource, model, classes)
        self.buffer = ['1', '2', '3']

    def createVideoWriter(self, player, outFile: str):
        xShape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        yShape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourCC = cv2.VideoWriter_fourcc(*"MJPG")

        return cv2.VideoWriter("detections\\" + outFile, fourCC, 20, (xShape, yShape))

    def waitForKeyPress(self):
        input("Press any key to stop the detection...")

    def detect(self, source: str, outFile: str):
        keyThread = threading.Thread(target=self.waitForKeyPress)
        keyThread.start()

        player = self.dataSource.loadData(source)
        out = self.createVideoWriter(player=player, outFile=outFile)

        fps = 0
        cFrame = 0
        while keyThread.is_alive():
            startTime = time()
            ret, frame = player.read()

            labels, cord = super().scoreFrame(frame=frame)
            plotter = Plotter(self.classes)

            if not ret:
                break

            endTime = time()
            fps = 1/np.round(endTime - startTime, 3)

            frame = plotter.plot(frame=frame, fps=fps,
                                 labels=labels, cords=cord)

            out.write(frame)
            self.writeFrameToBuffer(frame=frame, currentFrame=cFrame)
            cFrame += 1

        out.release()
        player.release()

    def writeFrameToBuffer(self, frame, currentFrame):
        cv2.imwrite("buffer\\buffer" + str(int(currentFrame % bufferSize)) + ".jpg", frame)