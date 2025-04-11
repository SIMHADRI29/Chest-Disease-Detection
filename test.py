import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model

cnn_model = load_model("model/cnn_weights.hdf5")
labels = []
for root, dirs, directory in os.walk('Dataset'):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())   
print("Chest Disease Class Labels : "+str(labels))

path = "testImages/Normal"
count = 0
for root, dirs, directory in os.walk(path):#connect to dataset folder
    for j in range(len(directory)):#loop all images from dataset folder
        img = cv2.imread(root+"/"+directory[j])#read images
        img = cv2.resize(img, (28, 28))#resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,3)#convert image as 4 dimension
        img = np.asarray(im2arr)
        img = img.astype('float32')#convert image features as float
        img = img/255 #normalized image
        predict = cnn_model.predict(img)#now predict dog breed
        predict = np.argmax(predict)
        print(labels[predict])
    print()    
    
            
