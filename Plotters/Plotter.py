import cv2
import numpy as np

class Plotter:
    def __init__(self, classes: dict):
        self.classes = classes

    def classToLabel(self, x):
        """
        Returns the label name in a string format from the corresponding numeric label.

        Parameters
        ----------
        x:
            The numeric label.
        """

        return self.classes[int(x)]

    def plotBoxes(self, frame, labels, cords):
        """
        Takes a frame and it's results as input and plots bounding boxes and labels onto the frame.

        Parameters
        ----------
        frame:
            Image frame on which the bounding boxes and labels will be plotted.
        """

        n = len(labels)
        xShape, yShape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cords[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * xShape), int(row[1] * yShape), int(row[2] * xShape), int(row[3] * yShape)

                r = (int(labels[i] + 1) * 11) % 255
                g = (int(labels[i] + 1) * 17) % 255
                b = (int(labels[i] + 1) * 13) % 255

                background = (r, g, b)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)
                cv2.putText(frame, self.classToLabel(labels[i]), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, "Detected: " + str(len(labels)), (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)

        return frame

    def displayFPS(self, frame, fps:float):
        colour = (255, 255, 255)
        text = "FPS: " + str(np.round(fps, 2))
        location = (20, 20)

        cv2.putText(
            frame,
            text,
            location,
            cv2.FONT_ITALIC,
            0.9,
            colour,
            2
        )

        return frame

    def plot(self, frame, fps, labels, cords):
        boxes = self.plotBoxes(frame=frame, labels=labels, cords=cords)
        frames = self.displayFPS(boxes, fps=fps)

        return frames