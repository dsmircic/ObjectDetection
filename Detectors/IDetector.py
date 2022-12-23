import abc

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter


class IDetector(abc.ABC):
    """
    Generic detector for object detection, doesn't really do anything unless it is extended and the "detect" method is overriden.
    """
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        self.dataSource = dataSource
        self.model = model
        self.classes = classes

    def scoreFrame(self, frame):
        """
        Takes a single frame as input, and scores it using the yolov5 model.

        Parameters
        ----------
        frame:
            The frame on which detection will be made.
        """

        frame = [frame]
        results = self.model(frame)

        labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coord

    @abc.abstractmethod
    def detect(self, source: str, outFile: str):
        """
        Runs a detection algorithm and plots the results to the screen.
        """
        pass
