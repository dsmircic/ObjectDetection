import cv2
from DataLoaders.IDataLoader import IDataLoader


class CameraLoader(IDataLoader):
    """
    Loads a video from the desired path in the appropriate cv2 format.
    """

    def load_data(self, path = None):
        return cv2.VideoCapture(0)

