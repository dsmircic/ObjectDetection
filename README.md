# Purpose
This program runs a Yolv5 detection algorithm and plots bounding boxes for detected objects to the specified output file.

---
## Usage
When starting the program from the terminal you can use:
    --classes or -c -> specifies classes of objects which you want to detect (0 - person, 1 - bicycle, 2 - car ...)
    --conf -> specifies the confidence level above which objects will be detected

All class names and label values are specified in the classes.txt file

### Notes
Program can run detection on YT videos, local videos and images without changes in code.
The only specification is in the "main" method where you need to provide a source for detection, and where the results are going to be saved. All results are located in the "detections" folder

In order to stop the detection on a video, the program needs to be started from the console, not from an editor such as VS Code.

---
## Problems
There are problems with loading .mp4 files.