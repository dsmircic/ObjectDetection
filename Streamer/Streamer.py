import cv2
import os

from time import time
from flask import Flask, Response

app=Flask(__name__)

def readVideo():
    while True:
        frame = open(os.getcwd() + 
                        "\\..\\buffer\\buffer.jpg", 'rb').read()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(readVideo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)