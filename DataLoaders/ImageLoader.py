import numpy as np
import cv2
from DataLoaders.IDataLoader import IDataLoader


class ImageLoader(IDataLoader):

    def loadData(self, path: str):
        return cv2.imread(path)
