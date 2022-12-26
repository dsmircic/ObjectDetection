import cv2
import os

from time import time
from flask import Flask, Response
from Detectors.VideoDetector import VideoDetector

app=Flask(__name__)

def read_video():
    while True:
        c = 0
        # frame = open(os.getcwd() + "\\..\\buffer\\buffer"
        #             + (str(int((c) % 1)) + ".jpg"), 'rb').read()
        frame = open(os.getcwd() + "\\..\\buffer\\buffer.jpg", 'rb').read()
        c += 1

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(read_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)