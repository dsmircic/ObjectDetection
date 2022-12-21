from Detectors.IDetector import IDetector

from DataLoaders.IDataLoader import IDataLoader
from Plotters.Plotter import Plotter

class ImageDetector(IDetector):
    def __init__(self, dataSource: IDataLoader, model, classes: dict):
        super().__init__(dataSource, model, classes)

    def detect(self, source: str, outFile: str):
        pass