import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten,Dropout,Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle


############################
path='train'
testRatio=0.2
validationRatio=0.2
imageDimensions=(32,32,3)

batch_size=64
epochs=15
steps_per_epoch=2000
##########################
mylist=os.listdir(path)
NoOfClasses=len(mylist)
images=[]
classno=[]

for x in range (0,NoOfClasses):
    myPicList=os.listdir(path +"/"+ str(x))
    for y in myPicList:
        CurImg=cv2.imread(path +"/"+ str(x)+"/"+y)
        CurImg=cv2.resize(CurImg,(32,32))
        images.append(CurImg)
        classno.append(x)
#print(x)
#print(len(images))
#print(len(classno))
images=np.array(images)
classno=np.array(classno)
#print(images.shape) 

      
######Spliting Data #### 
#######################
X_train,X_test,Y_train,Y_test=train_test_split(images,classno,test_size=testRatio) 
X_train,X_validation,Y_train,Y_validation= train_test_split(X_train,Y_train,test_size=validationRatio) 
#print(X_train.shape)
#print(X_test.shape)
#print(X_validation.shape) 
NoOfSamples=[]
#print((np.where(Y_train==0)))
for x in range (0,NoOfClasses):
   NoOfSamples.append(len(np.where(Y_train==x)[0]))
#print(NoOfSamples)     

#plt.figure(figsize=(10,5))
#plt.bar(range(0,NoOfClasses),NoOfSamples)
#plt.title("no of images for each class")
#plt.xlabel("Class id")
#plt.ylabel("no of images")
#plt.show()


###pre processing ##
#######################
def pre_processing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255  #normalise
    return img
#img=pre_processing(X_train[30])
#img=cv2.resize(img,(300,300))
#cv2.imshow("image",img)
#cv2.waitkey(0)

X_train=np.array(list(map(pre_processing,X_train)))
X_test=np.array(list(map(pre_processing,X_test)))
X_validation=np.array(list(map(pre_processing,X_validation)))


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

#### IMAGE AUGMENTATION 
dataGen= ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
dataGen.fit(X_train)
#### ONE HOT ENCODING OF MATRICES

Y_train=to_categorical(Y_train,NoOfClasses)
Y_test=to_categorical(Y_test,NoOfClasses)
Y_validation=to_categorical(Y_validation,NoOfClasses)

def my_model():
    NoOfFilters=60
    SizeOfFilter1=(5,5)
    SizeOfFilter2=(3,3)
    SizeOfPool=(2,2)
    NoOfNodes=500
    model=Sequential()
    model.add((Conv2D(NoOfFilters,SizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(NoOfFilters,SizeOfFilter1,activation='relu')))  
    model.add(MaxPooling2D(pool_size=SizeOfPool))
    model.add((Conv2D(NoOfFilters//2,SizeOfFilter2,activation='relu'))) 
    model.add((Conv2D(NoOfFilters//2,SizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=SizeOfPool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(NoOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43,activation='softmax')) 
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=my_model()
print(model.summary())



history=model.fit_generator(dataGen.flow(X_train,Y_train,
                                 batch_size=batch_size),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=(X_validation,Y_validation),
                                 shuffle=1)

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epochs')

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epochs')
plt.show()

score= model.evaluate(X_test,Y_test,verbose=0)
print('Test Score=',score[0])
print('Test Accuracy',score[1])

###save model
pickle_out=open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()





