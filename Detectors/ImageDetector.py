import cv2

from Detectors.IDetector import IDetector
from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter


class ImageDetector(IDetector):
    """
    Detects objects on images.
    """
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        super().__init__(dataSource, model, classes)

    def detect(self, source: str, outFile: str):
        frame = self.dataSource.loadData(source)
        data = super().scoreFrame(frame=frame)

        plotter = Plotter(self.classes)
        outFrame = plotter.plot(frame, None, labels=data["labels"], cords=data["cords"], confidence=data["confidence"])

        cv2.imwrite("detections\\" + outFile, outFrame)
        cv2.imshow("Detection", outFrame)
        cv2.waitKey(0)
