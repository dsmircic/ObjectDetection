import abc
import numpy as np

import torch

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

    def score_frame(self, frame) -> dict:
        """
        Takes a single frame as input, and scores it using the yolov5 model.

        Parameters
        ----------
        frame:
            The frame on which detection will be made.
        """
        data = dict()

        frame = [frame]
        results = self.model(frame)

        data["confidence"] = results.xyxy[0][0:, 4].numpy()
        data["labels"] = results.xyxyn[0][:, -1]
        data["coords"] = results.xyxyn[0][:, :-1]

        return data

    @abc.abstractmethod
    def detect(self, source: str, outFile: str):
        """
        Runs a detection algorithm and plots the results to the screen.
        """
        pass
