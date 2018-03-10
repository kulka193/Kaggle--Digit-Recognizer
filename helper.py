#import tensorflow as tf
import numpy as np
import pandas as pd
counter=0

def read_and_preprocess(filename):
    #data=np.genfromtxt(filename,delimiter=',')
    data=pd.read_csv('./'+filename)
    if filename=='train.csv':
        Y=data['label'].values
        Y=Y.astype(np.uint8)
        pix=data.iloc[:,1:].values
        X=pix.astype(np.float)
        X=(1/255.0)*X
        return X,Y
    elif filename=='test.csv':
        testimages=data.iloc[:,:].values
        X=testimages.astype(np.float)
        X=(1/255.0)*X
        return X

def getclasses(y):
    classes = np.unique(y)
    return classes

def dense_to_one_hot(y):
    classes=getclasses(y)
    n_classes=len(classes)
    num_train=y.shape[0]
    y_hot_encoded=np.zeros((num_train,n_classes))
    for i in range(num_train):
        for col in classes:
            if y[i]==col:
                y_hot_encoded[i,col]=1
    return y_hot_encoded

def one_hot_to_dense(y_hot_encoded):
    return np.argmax(y_hot_encoded,1)
    
def next_batch(X,y,batch_size,indices):
    global counter
    #global indices
    start=counter
    end=start+batch_size
    if counter%batch_size==0:
        np.random.shuffle(indices)
    counter+=1
    batchx=X[indices[start:end],:]
    batchy=y[indices[start:end],:]
    return batchx,batchy

