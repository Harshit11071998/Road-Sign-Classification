import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report,confusion_matrix
pickle_in=open("model_trained.p","rb")
model=pickle.load(pickle_in)



def pre_processing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255  #normalise
    return img

data=[]

y_test = pd.read_csv('Test.csv')
 
labels = y_test["ClassId"].values
imgs = y_test["Path"].values

for img in imgs:
    image=cv2.imread(img)
    image=cv2.resize(image,(32,32))
    image=pre_processing(image)
    
    data.append(np.array(image))
    
X_test=np.array(data)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
pred = model.predict_classes(X_test)
print(accuracy_score(labels, pred))

#y_pred=model.predict_classes(X_test)
#print(classification_report(labels,y_pred))
#
#print(confusion_matrix(labels,y_pred))

