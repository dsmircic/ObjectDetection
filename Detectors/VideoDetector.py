import threading
import cv2
import numpy as np

from Detectors.IDetector import IDetector
from time import time, sleep

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter


class VideoDetector(IDetector):
    """
    Detects objects on videos.
    """

    def __init__(self, dataSource: IDataLoader, model, params: dict, classes: dict):
        super().__init__(dataSource, model, classes)
        self.params = params

    def create_video_writer(self, player, out_file: str):
        xShape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        yShape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourCC = cv2.VideoWriter_fourcc(*"MJPG")

        return cv2.VideoWriter("detections\\" + out_file, fourCC, 20, (xShape, yShape))

    def wait_for_key_press(self):
        input("Press any key to stop the detection...")

    def get_fps(self, start_time, end_time) -> float:
        return self.params["speed"] / np.round(end_time - start_time, 2)

    def detect(self, source: str, out_file: str):
        key_thread = threading.Thread(target=self.wait_for_key_press)
        key_thread.start()

        player = self.dataSource.load_data(source)
        out = self.create_video_writer(player=player, out_file=out_file)

        fps = 0
        current_frame = 0
        plotter = Plotter(self.classes)

        while key_thread.is_alive():
            start_time = time()
            ret, frame = player.read()
            
            if not ret:
                break

            skip_frames = self.params["speed"]

            if current_frame % skip_frames == 0:
                data = super().score_frame(frame=frame)
                end_time = time()
                fps = self.get_fps(start_time=start_time, end_time=end_time)


            base = self.params["base"]
            overlap = self.params["overlap"]

            frame = plotter.plot(frame=frame, fps=fps,
                                 labels=data["labels"], cords=data["coords"], confidence=data["confidence"], base=base, overlap=overlap)

            current_frame += 1
            out.write(frame)

        out.release()
        player.release()
