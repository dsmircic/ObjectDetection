import cv2
import numpy as np

font = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.82
fontThickness = 2
size, _ = cv2.getTextSize('Test', font, fontScale=fontScale, thickness=fontThickness)
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

    def plot_boxes(self, frame, labels, cords, confidence):
        """
        Takes a frame and it's results as input and plots bounding boxes and labels onto the frame.

        Parameters
        ----------
        frame:
            Image frame on which the bounding boxes and labels will be plotted.
        """

        n = len(labels)
        xShape, yShape = frame.shape[1], frame.shape[0]

        if n == 0:
            text = "Detected: " + "0"
            cv2.putText(frame, text, (20, topMargin),
                    fontFace=font, fontScale=fontScale, color=white, thickness=fontThickness)

        for i in range(n):
            row = cords[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(
                    row[0] * xShape), int(row[1] * yShape), int(row[2] * xShape), int(row[3] * yShape)

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

    def plot(self, frame, fps, labels, cords, confidence):
        frames = self.plot_boxes(frame=frame, labels=labels, cords=cords, confidence=confidence)

        if fps is not None:
            frames = self.display_fps(frames, fps=fps)

        return frames
