import cv2
from DataLoaders.IDataLoader import IDataLoader

class VideoLoader(IDataLoader):

    def loadData(self, path: str):
        return cv2.VideoCapture(path)