import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2 as cv
# Parameters
input_size = (150,150) # Bisa kalian ganti
#define input shape
channel = (3,)
input_shape = input_size + channel
#define labels
labels = ['applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 'cooking','drinking' ,'jumping' , 'phoning' ,'playing_guitar' ,'playing_violin' , 'pouring_liquid', 'reading', 'running', 'smoking', 'washing_dishes','watching_TV', 'waving_hands', 'writing_book']

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr
def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

from tensorflow.keras.models import load_model
MODEL_PATH = 'my_model_adamax2.h5'
model = load_model(MODEL_PATH,compile=False)

cap = cv.VideoCapture(0)

while True:
    success,frame = cap.read()
    im = Image.fromarray(frame)
    X = preprocess(im,input_size)
    X = reshape([X])
    y = model.predict(X)
    print( labels[np.argmax(y)], np.max(y) )

    cv.imshow("window", frame)
    cv.waitKey(1)