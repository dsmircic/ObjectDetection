# Purpose
This program runs a Yolv5 detection algorithm and plots bounding boxes for detected objects to the specified output file.

---
## **1. Usage**
When starting the program from the terminal you can use:
- --classes or -c - specifies classes of objects which you want to detect (0 - person, 1 - bicycle, 2 - car ...)
- --conf - specifies the confidence level above which objects will be detected
- --source - specifies the source of the file where detection will be done
- --dest - specifies the name of the dest file where results will be stored (in the detections/ dir)

All class names and label values are specified in the `classes.txt` file

---
### **1.1 Notes**
Program can run detection on YT videos, local videos and images without any changes in code.
The only specification is in the "main" method where you need to provide a source for detection, and where the results are going to be saved. All results are located in the "detections" directory.

In order to stop the detection on a video, the program needs to be started from the console, not from an editor such as VS Code.

All requirements are specified in the `requirements.txt` file and can be installed with:

```bash
pip install -r requirements.txt
```

### **1.2 Examples**
#### bash
```bash
.\ObjectDetection.py --conf 0.3 --source <image_name>.png  -- dest <dest_file_name>.png --classes 0 1 2
```

```bash
.\ObjectDetection.py --conf 0.4 --source <yt_video_link> --dest <video>.avi --classes 3 4 12
```
---
Instead of passing these arguments through the command line, you 
can pass them inside the `ObjectDetector.py` program


### through code
```py
detector = ObjectDetector("source_name", "dest_name")
detector.detect()
```

Note that if you start detection through code, you cannot stop video detection with a press of a key on your keyboard. You can only do this through the terminal/command line.

---
## **3. Implementation**
This code was implemented on a Windows platform.

---
## **4. Problems**
There are problems with loading .mp4 files.

---
---
## *Author*
- [Dino Smirčić](https://github.com/dsmircic)