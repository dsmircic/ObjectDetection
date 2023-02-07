import cv2

from Detectors.IDetector import IDetector
from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter


class ImageDetector(IDetector):
    """
    Detects objects on images.
    """
    def __init__(self, dataSource: IDataLoader, model, params: dict, classes: dict):
        super().__init__(dataSource, model, classes)

    def detect(self, source: str, out_file: str):
        frame = self.dataSource.load_data(source)
        data = super().score_frame(frame=frame)

        plotter = Plotter(self.classes)
        out_frame = plotter.plot(frame, None, labels=data["labels"], cords=data["coords"], confidence=data["confidence"])

        cv2.imwrite("detections\\" + out_file, out_frame)
        cv2.imshow("Detection", out_frame)
        cv2.waitKey(0)
