import cv2 as cv
import numpy as np
import requests
from io import BytesIO
from PIL import Image


# Parameters
input_size = (150,150) # Bisa kalian ganti
#define input shape
channel = (3,)
input_shape = input_size + channel
#define labels
labels = ['applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 'cooking','drinking' ,'jumping' , 'phoning' ,'playing_guitar' ,'playing_violin' , 'pouring_liquid', 'reading', 'running', 'smoking', 'washing_dishes','watching_TV', 'waving_hands', 'writing_book']


from tensorflow.keras.models import load_model
MODEL_PATH = 'my_model_adamax2.h5'
model = load_model(MODEL_PATH,compile=False)


 
cap = cv.VideoCapture(1)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2
 
#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr
def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


def findObjects(outputs,img, capHeight, capWidth):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    crop_frame =[]
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int(((det[1]*hT)-h/2)-20)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                ynew = y+h
                xnew = x+w
                if x >= capWidth:
                    x = capWidth
                elif x <=0:
                    x=0
                if xnew >= capWidth:
                    xnew = capWidth
                elif xnew <=0:
                    xnew=0
                if y >= capHeight:
                    y = capHeight
                elif y <=0:
                    y=0
                if ynew >= capHeight:
                    ynew = capHeight
                elif xnew <=0:
                    xnew=0
                crop_frame.append(img[y:ynew, x:xnew])
                
    
    # if len(crop_frame) > 0:
    #     # print(len(crop_frame))  
    #     for croped in range(len(crop_frame)-1):
    #         cv.imshow("image crop " + str(croped), crop_frame[croped])
    #         cv.destroyWindow("image crop " + str(croped))
    #         #print(croped)
    
   
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if classIds[i] == 0 :
            im = Image.fromarray(crop_frame[i])
            cX = preprocess(im,input_size)
            cX = reshape([cX])
            cY = model.predict(cX)
            # cv.imshow("image crop " + str(i), crop_frame[i])
        
            cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            
            cv.putText(img,f'{labels[np.argmax(cY)]} {int(np.max(cY)*100)}%',
                      (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
            # cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
            #           (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
    
 
while True:
    success, img = cap.read()
    capHeight, capWidth, channel = img.shape
    
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img, capHeight, capWidth)
 
    cv.imshow('Image', img)
    
    cv.waitKey(1)