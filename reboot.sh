#!/bin/bash
cd ~/tensorrt_demos
python3 trt_yolo.py --usb 0 -m yolov4-tiny_GIX-416 --category_num=3&
python3 ~/logtime.py
