# SnapSort

## Introduction
SnapSort is a trash sorting assistant with the YOLO V4 model. It will take in a video feed from the camera and return visual and audio instructions to users. The project is designed to help people sort their waste properly before waste collection. We placed the device, Jetson Nano or GTX 1080 with a screen, on top of trash cans to guide people on sorting waste.

The project is built for the University of Washington GIX TECHIN 514 Hardware Software Lab 1 course, Microsoft Imagine Cup, and Alaska Environment Innovation Challenges. The code and dataset are collected and built by Yun Liu, Joey Wang, and me. 

The dataset reaches 71% MAP for 12 different categories on an 80/20 train/test split trained on the YOLO v4 object detection model.

![alt text](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v3-Darknet/blob/master/Annotation%202020-06-24%20151209.png?raw=true)

## What's new (2021.03.04)
- Add trained weights with Yolo V4, V4 Tiny model
- Optimized performance for Jetson Nano with Onnx and Tensor RT core

## Dataset
The dataset contains 4600 original images including images from Coco Dataset, Google open images v4, and images we collected by ourselves. Images are labeled into 12 classes manually following [Seattle Government's recycling rule](https://www.seattle.gov/utilities/services/recycling/recycle-at-home/where-does-it-go---flyer). After data augmentation, 80% of images are used for training and 20% of images are used for testing. The best training result we got is 71% MAP @ 50% confidence threshold.

## Prerequisites

### Installing Yolo V4
The Yolo V4 model needs to be installed before using our trained weight. The most Windows-friendly version I found is from [AlexAB](https://github.com/AlexeyAB/darknet). 

### Installing Python and dependencies
* [Python 3 and above](https://www.python.org/downloads/)
* OpenCV 3.4.0 for Python
* [Numpy](https://numpy.org/)
* Winsound

## Usage (Run with Darknet)
### Step 1: Download and unzip
Download and unzip all the files into the Yolo build file.
Ex. mine location address `C:\darknet-master\build\darknet\x64`.

### Step 2: select a weight to be used
Change the file location according to the weights you want to use at line 159~162 in `darknet_video-Ku.py`.
**Option 1.** 12 classes detection
```
configPath = "./backup/0311_files/yolov3_5l-GIX_0304_test.cfg"
weightPath = "./backup/0311_files/yolov3_5l-GIX_0326_best.weights"
metaPath = "./backup/0311_files/Aug.data"
```
**Option 2.** 3 classes detection(Recycle, Landfill, Compost)
```
configPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_test.cfg"
weightPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_best.weights"
metaPath = "./backup/0311_files/Aug_RLC.data"
```
**Option 3.** 12 classes with Yolo V2
```
configPath = "./backup/yolov2/yolov2_GIX_test.cfg"
weightPath = "./backup/yolov2/yolov2_GIX_best.weights"
metaPath = "./backup/0311_files/Aug.data"
```
**Option 4.** 12 classes with Yolo V4
```
configPath = "./backup/Yolo_V4&V4_Tiny/yolov4_GIX_best-416.cfg"
weightPath = "./backup/Yolo_V4&V4_Tiny/yolov4_GIX_best-416.weights"
metaPath = "./backup/0311_files/Aug.data"
```
**Option 5.** 12 classes with Yolo V4 Tiny (416*416)
```
configPath = "./backup/Yolo_V4&V4_Tiny/yolov4-tiny_GIX-416.cfg"
weightPath = "./backup/Yolo_V4&V4_Tiny/yolov4-tiny_GIX-416.weights"
metaPath = "./backup/0311_files/Aug.data"
```
**Option 6.** 12 classes with Yolo V4 Tiny (640*640)
```
configPath = "./backup/Yolo_V4&V4_Tiny/yolov4-tiny_GIX-640.cfg"
weightPath = "./backup/Yolo_V4&V4_Tiny/yolov4-tiny_GIX-640.weights"
metaPath = "./backup/0311_files/Aug.data"
```

### Step 3: change the interval of audio instruction
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

![alt text](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v3-Darknet/blob/master/Annotation%202020-06-24%20151501.png?raw=true)

## Run Snapsort on Jetson Nano
### Step 1: Format SD card and install Jetpack
Formart a 32GB SD card(32GB or above) and install Jetpack 4.5.1

[Tutorial on Nvidia.com](https://developer.nvidia.com/embedded/jetpack)

### Step 2: Install Onnx and TensorRT
- Make sure connected to the internet (using an ethernet cable or wifi dongle)
- Open Command Line terminal
```
sudo apt-get install update
sudo apt-get install upgrade
sudo apt install python3-pip
pip3 install protobuf==3.8.0
git clone https://github.com/jkjung-avt/tensorrt_demos.git
cd tensorrt_demos/ssd
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
./install_pycuda.sh
sudo apt-get install python-pip protobuf-compiler libprotoc-dev
pip install Cython
sudo pip3 install onnx==1.4.1
cd ..
cd plugins
Make
```
- Copy all the files in the "Onnx & TensorRT' folder and paste them to folder “tensorrt_demos/yolo”

### Step 4:Change Category mapping
- Download and replace [yolo_classes.py](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v4-Darknet-Onnx-TensorRT-for-Jetson-Nano/blob/master/yolo_classes.py) in tensorrt_demos/utils
**Changes:**
	1. Line 6-21: Replace COCO_CLASSES_LIST with GIX_CLASSES_LIST & GIX_3CLASSES_LIST
	2. Line 36-39: Add parameter support: category_num=12 & category_num=3

- Download and replace [visualization.py](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v4-Darknet-Onnx-TensorRT-for-Jetson-Nano/blob/master/visualization.py) in tensorrt_demos/utils
**Changes:**
	1. Line 79: Add color_rule list to map with the color of trash bins
	2. Line 99-102: Add rules to apply to 3 classes and 12 classes

**Mapping rule**

| 12 classes               | 3 classes |
| ------------------------ |---------- |
| Plastic_soft             | Landfill  |
| Food_soiled_coated_paper | Landfill  | 
| Plastic_utensils         | Landfill  |
| Foam                     | Landfill  |
| Miscellaneous            | Landfill  |
| Carboard                 | Recycle   |
| Paper                    | Recycle   |
| Plastic_rigid            | Recycle   |
| Glass                    | Recycle   |
| Metal                    | Recycle   |
| Food_scrapes             | Compost   |
| Paper_towels             | Compost   |


### Step 3:Run testing
- Open Command Line terminal
- Plug-in a USB webcam

**Option 1.** 12 classes detection with Yolo V4 (416x416)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4_GIX_best-416 --category_num=12
```
**Option 2.** 12 classes detection with Yolo V4 Tiny (416x416)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-tiny_GIX-416 --category_num=12
```
**Option 3.** 12 classes detection with Yolo V4 Tiny (640x640)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-tiny_GIX-640 --category_num=12
```
**Option 4.** 3 classes detection with Yolo V4 (416x416)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4_GIX_best-416 --category_num=3
```
**Option 5.** 3 classes detection with Yolo V4 Tiny (416x416)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-tiny_GIX-416 --category_num=3
```
**Option 6.** 3 classes detection with Yolo V4 Tiny (640x640)
```
cd tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-tiny_GIX-640 --category_num=3
```
- "F" for fullscreen

### (Optional) Step 4: Auto fullscreen
- Double click /tensorrt_demos/trt_yolo.py to edit
- Add this line to Line #69 (Leave 6 blank spaces for indention)
```
        set_display(WINDOW_NAME, True)
```
- Save the file

### (Optional) Step 5: Auto-start after Boot
- Download [reboot.sh](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v4-Darknet-Onnx-TensorRT-for-Jetson-Nano/blob/master/reboot.sh) to ~/ （“Home” folder）
- Click the start button on the upper left corner, search and open “Start-up applications”. 
- On the “Start-up preference” window, click “Add” 
- Name: (any)
- Command: (select [reboot.sh](https://github.com/Kuchunan/SnapSort-Trash-Classification-with-YOLO-v4-Darknet-Onnx-TensorRT-for-Jetson-Nano/blob/master/reboot.sh))
- Click “Add”

	[Reference](https://itsfoss.com/manage-startup-applications-ubuntu/)


## Built With
- [Yolo v3 (original author)](https://pjreddie.com/)
- [AlexAB](https://github.com/AlexeyAB/darknet)
- [jkjung-avt](https://github.com/jkjung-avt/tensorrt_demos)

## Co-Author
- [Joey Wang](https://github.com/JoeyWangTW)
- [Yun Liu](https://github.com/yunliu61)

## Inspiration and Acknowledgments
- [Trashnet (inspired by)](https://github.com/garythung/trashnet)
- [NMS](https://www.kdnuggets.com/2019/12/pedestrian-detection-non-maximum-suppression-algorithm.html)
- [Labeling image tool](https://github.com/tzutalin/labelImg)
- [Global Innovation Exchange, University of Washington](https://gixnetwork.org/)

## Video Documentary
- [The Story of SnapSort - AI trash sorting assistant](https://www.youtube.com/watch?v=aTaK2tVGCpw&feature=youtu.be)
