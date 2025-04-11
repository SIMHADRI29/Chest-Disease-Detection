#import required classes and packages
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

main = tkinter.Tk()
main.title("Chest Disease Detection using CNN AI Algorithm") #designing main screen
main.geometry("1300x1200")

global X,Y,path,labels,cnn_model
global filename, dataset, labels, X_train, Y_train, text,x_test,y_test,y_train,y_test1

#define and load class labels found in dataset
path = "Dataset"
labels = []
X = []
Y = []
#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())   
print("Chest Disease Class Labels : "+str(labels))

#define function to get class label of given image
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index


def Upload():
    global X,Y,path,labels
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    #load dataset image and process them
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else: #if images not process then read and process image pixels
        for root, dirs, directory in os.walk(path):#connect to dataset folder
            for j in range(len(directory)):#loop all images from dataset folder
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])#read images
                    img = cv2.resize(img, (32, 32))#resize image
                    X.append(img) #add image pixels to X array
                    label = getLabel(name)#get image label id
                    Y.append(label)#add image label                
        X = np.asarray(X)#convert array as numpy array
        Y = np.asarray(Y)
        np.save('model/X.txt',X)#save process images and labels
        np.save('model/Y.txt',Y)
    text.insert(END,"Dataset images are loaded"+"\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Chest Disease Class Labels : ['Fibrosis', 'InflammationPenumonia', 'Normal', 'Pneumonia']")
    #visualizing class labels count found in dataset
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (6, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()
def preprocess():
    global X,Y,path,labels
    global filename, dataset, labels, X_train, Y_train, text,y_train,X_test,y_test,y_test1
    text.delete('1.0', END)
    global filename
    global X, Y
    #preprocess images like shuffling and normalization
    X = X.astype('float32')
    X = X/255 #normalized pixel values between 0 and 1
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle all images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Image Processing & Normalization Completed"+"\n")
    text.insert(END,"80% images used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% image used to test algorithms : "+str(X_test.shape[0])+"\n")
    
def calculateMetrics(algorithm, predict, y_test):
    global X,Y,path,labels
    global filename, dataset, labels, X_train, Y_train, text,y_train,X_test,y_test1
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 
    
    
def TrainCNN():
    global X,Y,path,labels,cnn_model
    global filename, dataset, labels, X_train, Y_train, text,x_test,y_test,y_train,X_test,y_test,y_test1
    #train CNN algorithm
    text.delete('1.0', END)
    cnn_model = Sequential()
    #defining cnn layer with 3 X 3 matrix to filter dataset using 32 neurons
    cnn_model.add(Convolution2D(64, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compiling, training and loading model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 64, epochs = 100, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test data   
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN Algorithm", predict, y_test1)#calculate accuracy and other metrics



def TrainRF():
    global X,Y,path,labels,cnn_model
    global filename, dataset, labels, X_train, Y_train, text,x_test,y_test,y_train,X_test,y_test,y_test1
    #train CNN algorithm
    text.delete('1.0', END)
    #training Random Forest machine learning algorithm
    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)
    X_train = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
    X_test = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest Algorithm", predict, y_test1)#calculate accuracy and other metrics
def TrainDT():
    global X,Y,path,labels,cnn_model
    global filename, dataset, labels, X_train, Y_train, text,x_test,y_test,y_train,X_test,y_test,y_test1
    #train CNN algorithm
    text.delete('1.0', END)
    #training decision tree algorithm
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("Decision Tree Algorithm", predict, y_test1)#calculate accuracy and other metrics
def graph():
    #plot all algorithm performance in tabular format
    import pandas as pd
    df = pd.DataFrame([['CNN Algorithm','Accuracy',accuracy[0]],['CNN Algorithm','Precision',precision[0]],['CNN Algorithm','Recall',recall[0]],['CNN Algorithm','FSCORE',fscore[0]],
                       ['Random Forest','Accuracy',accuracy[1]],['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','FSCORE',fscore[1]],
                       ['Decision Tree','Accuracy',accuracy[2]],['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()
    
def predict():
    global filename, dataset,cnn_model,np,labels
    text.delete('1.0', END)
    labels=['Fibrosis', 'InflammationPenumonia', 'Normal', 'Pneumonia']
    filename=filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)#read test image
    img = cv2.resize(image, (28, 28))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    predict = cnn_model.predict(img)#now predict dog breed
    predict = np.argmax(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,300))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Chest Disease Predicted As: '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    plt.figure(figsize=(12,12))
    cv2.imshow( 'Chest Disease Predicted As: '+labels[predict], img)
    cv2.waitKey(0)



def Exit():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text="Chest Disease Detection using CNN AI Algorithm") #designing main screen')
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=115)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=320,y=100)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset",command=Upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocessing Dataset",command=preprocess)
processButton.place(x=10,y=150)
processButton.config(font=font1)

cnnButton1 = Button(main, text="Run CNN Algorithm",command=TrainCNN)
cnnButton1.place(x=10,y=200)
cnnButton1.config(font=font1)

cnnButton = Button(main, text="Run Random Forest Algorithm",command=TrainRF)
cnnButton.place(x=10,y=250)
cnnButton.config(font=font1)


ensembleButton = Button(main, text="Run Decision Tree Algorithm",command=TrainDT)
ensembleButton.place(x=10,y=300)
ensembleButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph",command=graph)
graphButton.place(x=10,y=350)
graphButton.config(font=font1)


predictButton = Button(main, text="Predict from Test Image",command=predict)
predictButton.place(x=10,y=400)
predictButton.config(font=font1)

ExitButton = Button(main, text="Exit", command=Exit)
ExitButton.place(x=10,y=450)
ExitButton.config(font=font1)

main.config(bg='navy blue')
main.mainloop()
