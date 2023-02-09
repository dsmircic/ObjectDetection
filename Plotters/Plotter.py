import cv2
import numpy as np
import tensorflow as tf
from Helpers.Coordinates import Coordinates

font = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.82
fontThickness = 2
size, _ = cv2.getTextSize(
    'Test', font, fontScale=fontScale, thickness=fontThickness)
fontWidth, fontHeight = size
margin = 15
topMargin = 25
white = (255, 255, 255)


class Plotter:
    """
    Plots bounding boxes on images, prints the number of detected objects in the upper left corner and fps if the source file is a video.
    """

    def __init__(self, classes: dict):
        self.classes = classes

    def class_to_label(self, x):
        """
        Returns the label name in a string format from the corresponding numeric label.

        Parameters
        ----------
        x:
            The numeric label.
        """

        return self.classes[int(x)]

    def calculate_area(self, x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x2 - x1) * abs(y2 - y1)

    def check_object_cords_in_base_cords(self, base_cords: Coordinates, obj_cords: Coordinates) -> bool:
        """
        Checks if there is an object present inside the base object, returns True if there is, False otherwise.

        Parameters
        ----------
        base_cords:
            Coordinates object which contains x and y coordinates of the base object if it is present.
        
        obj_cords:
            Coordinate object which contains x and y coordinates of the object which is presumably inside the base.
        """
        if ((obj_cords.x1 >= base_cords.x1 and obj_cords.x1 <= base_cords.x2) or
            (obj_cords.x2 >= base_cords.x1 and obj_cords.x2 <= base_cords.x2)) and \
            ((obj_cords.y1 >= base_cords.y1 and obj_cords.y1 <= base_cords.y2) or
             (obj_cords.y2 >= base_cords.y1 and obj_cords.y2 <= base_cords.y2)):

            return True

        return False

    def find_base_cords(self, labels, base, xShape, yShape, cords) -> Coordinates:
        """
        Finds the coordinates of the base object which needs to be detected.

        Parameters
        ----------
        labels:
            Tensor object containing a list of detected object labels.
        
        base:
            Object which has to be present in order to detect other objects on top of it. It is set to -1 if there is no base object.
        
        xShape:
            Scalar which needs to be multiplied with every x coordinate in order to get Cartesian coordinates.

        yShape:
            Scalar which needs to be multiplied with every y coordinate in order to get Cartesian coordinates.
        
        cords:
            Tensor object containing a list of detected object coordinates.

        """
        if base in labels and base != -1:
                    cond = tf.equal(labels, base)
                    base_index = tf.keras.backend.eval(tf.where(cond))[0]
                    base_cords_yolo_format = cords[base_index].numpy()[0]
                    return Coordinates(int(base_cords_yolo_format[0] * xShape), int(base_cords_yolo_format[1] * yShape),
                                                    int(base_cords_yolo_format[2] * xShape), int(base_cords_yolo_format[3] * yShape))

        return None

    def plot_boxes(self, frame, labels, cords, confidence, base, overlap):
        """
        Takes a frame and it's results as input and plots bounding boxes and labels onto the frame.

        Parameters
        ----------
        frame:
            Image frame on which the bounding boxes and labels will be plotted.

        labels:
            Tensor object containing a list of detected object labels.

        cords:
            Tensor object containing a list of detected object coordinates.

        confidence:
            List of confidence levels for each detected object.
        
        base:
            Object which has to be present in order to detect other objects on top of it. It is set to -1 if there is no base object.
        """

        n = len(labels)
        xShape, yShape = frame.shape[1], frame.shape[0]

        if n == 0:
            text = "Detected: " + "0"
            cv2.putText(frame, text, (20, topMargin),
                        fontFace=font, fontScale=fontScale, color=white, thickness=fontThickness)

        base_cords = self.find_base_cords(labels=labels, base=base, xShape=xShape, yShape=yShape, cords=cords)
        
        for i in range(n):
            row = cords[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * xShape), int(row[1] * yShape), int(row[2] * xShape), int(row[3] * yShape)

                obj_cords = None
                if labels[i] != base and base != -1:
                    obj_cords = Coordinates(x1, y1, x2, y2)

                intersect = False
                if obj_cords is not None and base_cords is not None:
                    intersect = self.check_object_cords_in_base_cords(
                        base_cords, obj_cords)

                if labels[i] == base or intersect or base == -1:
                    r = (int(labels[i] + 1) * 11) % 255
                    g = (int(labels[i] + 1) * 13 - 120) % 255
                    b = (int(labels[i] + 1) * 17 - 25) % 255

                    background = (r, g, b)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)

                    cv2.putText(frame, self.class_to_label(
                        labels[i]), (x1, y1), fontFace=font, fontScale=fontScale, color=white, thickness=fontThickness)

                    cv2.putText(frame, str(np.round(confidence[i], 3)), (x2 - fontWidth, y1), fontFace=font,
                                fontScale=fontScale/1.5, color=white, thickness=fontThickness)

                    detected = str(len(labels)) if len(labels) > 0 else "0"

                    text = "Detected: " + detected
                    cv2.putText(frame, text, (20, topMargin),
                                fontFace=font, fontScale=fontScale, color=white, thickness=fontThickness)

        return frame

    def display_fps(self, frame, fps: float):
        text = "FPS: " + str(np.round(fps, 2))
        location = (20, fontHeight + topMargin + margin)

        cv2.putText(frame, text, location, color=white,
                    fontFace=font, fontScale=fontScale, thickness=fontThickness)

        return frame

    def plot(self, frame, fps, labels, cords, confidence, base, overlap):
        frames = self.plot_boxes(frame=frame, labels=labels, cords=cords,
                                 confidence=confidence, base=base, overlap=overlap)

        if fps is not None:
            frames = self.display_fps(frames, fps=fps)

        return frames
