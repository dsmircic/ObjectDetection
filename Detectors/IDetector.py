from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter

class IDetector:
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        self.dataSource = dataSource
        self.model = model
        self.classes = classes

    def detect(self, source: str, outFile: str):
        """
        Runs a detection algorithm and plots the results to the screen.
        """
        pass