# SnapSort
SnapSort is a trash sorting assistant with the YOLO V3 model. The project is built for the University of Washington GIX TECHIN 514 Hardware Software Lab 1 course, Microsoft Imagine Cup, and Alaska Environment Innovation Challenges. Code and dataset are collected and built by Yun Liu, [Joey Wang](https://github.com/JoeyWangTW) and I. The dataset reaches 63% MAP for 12 different categories on an 80/20 train/test split trained on YOLO v3 object detection model.

## Dataset
The dataset contains XXXX original images including images from Coco Dataset, Google open images v4, and images we collected by ourselves. Images are labeled into 12 classes manually following [Seattle Government's recycling rule](https://www.seattle.gov/utilities/services/recycling/recycle-at-home/where-does-it-go---flyer). After data augmentation, XXXX images are used for training and XXXX images are used for testing. The best training result we got is 63% MAP @ 50% confidence threshold.

## Prerequisites

### Intalling Yolo V3
The Yolo V3 model needs to be installed before using our trained weight. The most Windows-friendly version I found is from [AlexAB](https://github.com/AlexeyAB/darknet). 

### Installing Python and dependencies
* [Python 3 and above](https://www.python.org/downloads/)
* OpenCV 3.4.0 for Python
* [Numpy](https://numpy.org/)
* Winsound

## Usage
### Step 1: download and unzip
Download and unzip all the files into the Yolo build file.
Ex. mine location address `C:\darknet-master\build\darknet\x64`.

### Step 2: select a weight to be used
Change the file location according to the weights you want to use on line 159~162 in `darknet_video-Ku.py`
1. **12 classes detection**
```
configPath = "./backup/0311_files/yolov3_5l-GIX_0304_test.cfg"
weightPath = "./backup/0311_files/yolov3_5l-GIX_0326_best.weights"
metaPath = "./backup/0311_files/Aug.data"
```
2. **3 classes detection**(Recycle, Landfill, Compost)
```
configPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_test.cfg"
weightPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_best.weights"
metaPath = "./backup/0311_files/Aug_RLC.data"
```
3. **12 classes with Yolo V2**
```
configPath = "./backup/yolov2/yolov2_GIX_test.cfg"
weightPath = "./backup/yolov2/yolov2_GIX_best.weights"
metaPath = "./backup/0311_files/Aug_RLC.data"
```

### Step 3:Change the interval of audio instruction
Audio instruction will be given every 45 frames(default). It will collect all the detection results in these frames, merge the result for IoU higher than 50%, and give the audio feedback of the highest possibility. You can change the interval on line 14.
```
AvergeOverTime = 45
```
You can change the IOU on line 71.
```
def nms(detections,threshold=.5):
```

### Step 4: other minor changes
To change video quality:
```
VWidth = 1280
VHight = 720
```

To change input camera (0 means built-in camera)
```
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

To change thredhold:
```
detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.2,hier_thresh=0.3, nms=0.4) 
```

### Step 5: run the code and start detection
Run the code and the detection will start. Hit `ESC` to exit.

## Built With
- [Yolo v3 (original author)](https://pjreddie.com/)
- [AlexAB](https://github.com/AlexeyAB/darknet)

## Co-Author
- [Joey Wang](https://github.com/JoeyWangTW)
- [Yun Liu]

## Inspiration and Acknowledgments
- [trashnet (inspired by)](https://github.com/garythung/trashnet)
- [Nms](https://www.kdnuggets.com/2019/12/pedestrian-detection-non-maximum-suppression-algorithm.html)
