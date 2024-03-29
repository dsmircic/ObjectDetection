# Purpose
This program runs a Yolov5 detection algorithm and plots bounding boxes for detected objects to the specified output file.

---
## **1. Usage**
When starting the program from the terminal you can use:
- --classes or -c - specifies classes of objects which you want to detect (0 - person, 1 - bicycle, 2 - car ...)
- --conf - specifies the confidence level above which objects will be detected
- --source - specifies the source of the file where detection will be done
- --dest - specifies the name of the dest file where results will be stored (in the detections/ dir)
- --speed - specifies the speed of the detection, default is 5, max recommended is 10
- --base - specifies the object which needs to ge present in order to detect other objects on top of it

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

Note: some requirements are not needed, requirements.txt is a bit not up to date.

### **1.2 Examples**
#### bash
```bash
.\ObjectDetection.py --conf 0.3 --source <image_name>.png  -- dest <dest_file_name>.png --classes 0 1 2 --conf 0.5 --speed 3 --base 63
```

```bash
.\ObjectDetection.py --conf 0.4 --source <yt_video_link> --dest <video>.avi --classes 3 4 12 --conf 0.5 --speed 3 --base 63
```
---
## **2. Implementation**
This code was implemented on a Windows platform.

---
## **3. Problems**
There are problems with loading some videos from Youtube, because of the newly updated "likes column" on Youtube.

---
---
## *Author*
- [Dino Smirčić](https://github.com/dsmircic)