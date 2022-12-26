import numpy as np
import cv2
from DataLoaders.IDataLoader import IDataLoader


class ImageLoader(IDataLoader):
    """
    Loads an image from the desired path in the appropriate cv2 format.
    """

    def load_data(self, path: str):
        return cv2.imread(path)
