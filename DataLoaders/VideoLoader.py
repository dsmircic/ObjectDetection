import cv2
from DataLoaders.IDataLoader import IDataLoader


class VideoLoader(IDataLoader):
    """
    Loads a video from the desired path in the appropriate cv2 format.
    """

    def load_data(self, path: str):
        return cv2.VideoCapture(path)
