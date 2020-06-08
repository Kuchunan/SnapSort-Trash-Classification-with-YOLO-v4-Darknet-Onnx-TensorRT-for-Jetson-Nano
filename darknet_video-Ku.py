from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import winsound

VWidth = 1280
VHight = 720
Rule = {'Person':'Person','Plastic_soft':'Landfill','Food_soiled_coated_paper':'Landfill','Plastic_utensils':'Landfill','Foam':'Landfill','Miscellaneous':'Landfill','Cardboard':'Recycle','Paper':'Recycle','Plastic_rigid':'Recycle','Glass':'Recycle','Metal':'Recycle','Food_scraps':'Compost','Paper_towels':'Compost','GIX_utensils':'Compost','Recycle':'Recycle','Landfill':'Landfill','Compost':'Compost'}
AvergeOverTime = 45


def convertBack(x, y, w, h,darknet_width, darknet_hight):
    x = darknet.network_width(netMain) -x #Mirroring adjustment
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    xmin = int(xmin*(VWidth/darknet_width))
    ymin = int(ymin*(VHight/darknet_hight))
    xmax = int(xmax*(VWidth/darknet_width))
    ymax = int(ymax*(VHight/darknet_hight))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img,darknet_width, darknet_hight):
    for detection in detections:
        x, y, w, h = detection[2][0],detection[2][1],detection[2][2],detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h),darknet_width, darknet_hight)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        ymid = int((ymin+ymax)/2)
        ClassNme = Rule[str(detection[0].decode())]
        print(str(detection[0].decode())+" "+str(round(detection[1] * 100, 2)))
        if ClassNme =='Landfill':
            ClassNme = ClassNme.upper()          
            cv2.rectangle(img, pt1, pt2, (128, 128, 128), 5)
            (text_width, text_height) = cv2.getTextSize(ClassNme, cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2)[0]
            cv2.rectangle(img, (xmin, ymid),(xmin+text_width,ymid-text_height), [128, 128, 128], cv2.FILLED)
            cv2.putText(img,
                        ClassNme,
                        (pt1[0], ymid), cv2.FONT_HERSHEY_DUPLEX, 2,
                        [255, 255, 255], 2) #'''+" [" + str(round(detection[1] * 100, 2)) + "]"'''
        elif ClassNme =='Recycle': 
            ClassNme = ClassNme.upper()          
            cv2.rectangle(img, pt1, pt2, (0, 75, 141), 5)
            (text_width, text_height) = cv2.getTextSize(ClassNme, cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2)[0]
            cv2.rectangle(img, (xmin, ymid),(xmin+text_width,ymid-text_height), [0, 75, 141], cv2.FILLED)
            cv2.putText(img,
                        ClassNme,
                        (pt1[0], ymid), cv2.FONT_HERSHEY_DUPLEX, 2,
                        [255, 255, 255], 2)

        elif ClassNme =='Compost': 
            ClassNme = ClassNme.upper()          
            cv2.rectangle(img, pt1, pt2, (121, 156, 75), 5)
            (text_width, text_height) = cv2.getTextSize(ClassNme, cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2)[0]
            cv2.rectangle(img, (xmin, ymid),(xmin+text_width,ymid-text_height), [79, 121, 66], cv2.FILLED)
            cv2.putText(img,
                        ClassNme,
                        (pt1[0], ymid), cv2.FONT_HERSHEY_DUPLEX, 2,
                        [255, 255, 255], 2)
    return img




def nms(detections,threshold=.5):
    def overlapping_area(detection_1, detection_2):
        x1_tl = detection_1[2][0]
        x2_tl = detection_2[2][0]
        x1_br = detection_1[2][0] + detection_1[2][2]
        x2_br = detection_2[2][0] + detection_2[2][2]
        y1_tl = detection_1[2][1]
        y2_tl = detection_2[2][1]
        y1_br = detection_1[2][1] + detection_1[2][3]
        y2_br = detection_2[2][1] + detection_2[2][3]
        # Calculate the overlapping Area
        x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
        y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
        overlap_area = x_overlap * y_overlap
        area_1 = detection_1[2][2] * detection_1[2][3]
        area_2 = detection_2[2][2] * detection_2[2][3]
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / float(total_area) 
    detections = sorted(detections, key=lambda detections: detections[1],reverse=True)
    # for i in range(len(detections):
    #     if float(detections[i][1]) <0.2:
    #         del detections[i]
    new_detections=[]   
    new_detections.append(detections[0])
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) < threshold : #and str(detection[0].decode())== str(new_detection[0].decode())
                new_detections.append(detection)
                
    new_detections = sorted(new_detections, key=lambda detections: detections[1],reverse=True)
    return new_detections

def Dropperson(detections):
    for detection in detections:
        if str(detection[0].decode())=='Person':
            detections.remove(detection)
            Dropperson(detections)
    return detections        

def UIBar(detections,img,darknet_width, darknet_hight):
    def CropCombine(frame,BackGround,xmin, ymin, xmax, ymax,width,hight,centerX,centerY):
        imgcrop = frame[ymin:ymax,xmin:xmax]
        imgcrop = cv2.resize(imgcrop, (width,hight))
        BackGround[centerX:centerX+width,centerY:centerY+hight] = imgcrop[:,:]
        return BackGround
    
    if len(detections)>0:
        imgUI = cv2.vconcat([imgInstruction])
        for order in range(0,3):
            try:
                detection = detections[order]
                ClassNme = Rule[str(detection[0].decode())]
                x, y, w, h = detection[2][0],detection[2][1],detection[2][2],detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h),darknet_width, darknet_hight)
                #print(str(detection[0].decode()))
                if ClassNme =='Landfill':
                    imgUI = cv2.vconcat([imgUI,cv2.cvtColor(CropCombine(img,imgLandfillInstruction,xmin, ymin, xmax, ymax,150,150,60,35), cv2.COLOR_BGR2RGB)])
                elif ClassNme =='Recycle':
                    imgUI = cv2.vconcat([imgUI,cv2.cvtColor(CropCombine(img,imgRecycleInstruction,xmin, ymin, xmax, ymax,150,150,60,35), cv2.COLOR_BGR2RGB)])
                elif ClassNme =='Compost':
                    imgUI = cv2.vconcat([imgUI,cv2.cvtColor(CropCombine(img,imgCompostInstruction,xmin, ymin, xmax, ymax,150,150,60,35), cv2.COLOR_BGR2RGB)])
            except:
                imgUI = cv2.vconcat([imgUI,imgBlankInstruction])
                
        #imgUI = cv2.vconcat([imgInstruction,imgLandfillInstruction,imgCompostInstruction,imgRecycleInstruction])  
    elif len(detections)==0:
        imgUI = cv2.vconcat([imgDetecting,imgLandfillLogo,imgCompostLogo,imgRecycleLogo])
    
    ###rescale
    # scale_percent = (imgUI.shape[0] / 720)
    # width = int(imgUI.shape[1] / scale_percent)
    # height = int(imgUI.shape[0] / scale_percent)
    # imgUI=cv2.resize(imgUI,(width, height))
    return imgUI

def SoundFeedback(detection):
    playfile = 'SoundFeedback/' +str(Rule[str(detection[0][0].decode())]) + '.wav'
    winsound.PlaySound(playfile, winsound.SND_FILENAME | winsound.SND_ASYNC)  

netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    #12 Category
    configPath = "./backup/0311_files/yolov3_5l-GIX_0304_test.cfg"
    weightPath = "./backup/0311_files/yolov3_5l-GIX_0326_best.weights"
    metaPath = "./backup/0311_files/Aug.data"


    #RLC
    # configPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_test.cfg"
    # weightPath = "./backup/0311_files/yolov3_5l-GIX_0308_RLC_best.weights"
    # metaPath = "./backup/0311_files/Aug_RLC.data"

    #YoloV2
    # configPath = "./backup/yolov2/yolov2_GIX_test.cfg"
    # weightPath = "./backup/yolov2/yolov2_GIX_best.weights"
    # metaPath = "./backup/0311_files/Aug.data"


    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture("test.mp4")
    #cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    cap.set(3, VWidth)
    cap.set(4, VHight)
    #cap.set(cv2.CAP_PROP_FPS, fps)
    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(1280, 720))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),darknet.network_height(netMain),3)
    OTDetection = list()
    imgUI = cv2.imread('UI/Logo.png')
    imgBottom = cv2.imread('UI/Bottom.png') 
    frameCounter =0
    prev_time = time.time()


    OvertimeCounter = 0
    while True:
        ret, frame_read = cap.read()
        #frame_read=frame_read[:VHight,:VWidth,:]
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,(darknet.network_width(netMain),darknet.network_height(netMain)),interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.2,hier_thresh=0.3, nms=0.4)  
        frame_rgb = cv2.flip(frame_rgb,1)
        frame_rgb1 = frame_rgb.copy()
        image = cvDrawBoxes(detections, frame_rgb,darknet.network_width(netMain),darknet.network_height(netMain))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.hconcat([image, imgUI])
        #image = cv2.vconcat([image, imgBottom])
        OTDetection+=detections
        OvertimeCounter+=1
        if OvertimeCounter ==AvergeOverTime:
            OvertimeCounter = 0
            #OTDetection = Dropperson(OTDetection)
            print(frameCounter) 
            if len(OTDetection) >0:
                #OTDetection = nms(OTDetection, threshold=.1)
                SoundFeedback(sorted(OTDetection, key=lambda detections: detections[1],reverse=True))
            #     print(len(OTDetection))
            #     imgUI = UIBar(OTDetection,frame_rgb1,darknet.network_width(netMain),darknet.network_height(netMain))
            # elif len(OTDetection) ==0:
            #     imgUI = cv2.vconcat([imgDetecting,imgLandfillLogo,imgCompostLogo,imgRecycleLogo])
            OTDetection.clear()
            

        cv2.imshow('SnapSort!', image)
        frameCounter+=1
        print("fps ", frameCounter/(time.time()-prev_time))

        k = cv2.waitKey(3)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    cap.release()
    #out.release()

if __name__ == "__main__":
    YOLO()
