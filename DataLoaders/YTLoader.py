from DataLoaders.IDataLoader import IDataLoader
import pafy
import cv2

class YTLoader(IDataLoader):

    def loadData(self, path: str):
        """Loads the video from the URL in the lowest available resolution"""
        video = pafy.new(path).streams[-1]
        assert video is not None
        return cv2.VideoCapture(video.url)