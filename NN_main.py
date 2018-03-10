# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
#print(check_output(["ls","../input"]).decode("utf8"))
#import matplotlib.pyplot as plt
#% matplotlib inline
import os
import helper
import imshow

image_size=28
n_classes = 10
batch_size = 64
alpha=1e-4
dr=0.8  #decayrate
input_channels=1
stride_size=2
filters=[1,32,28]
patch=[5,5]
flattened_layer=7*7*filters[-1]  
fc=[flattened_layer,400,120,80,10]
x = tf.placeholder('float', [None, image_size*image_size])
y = tf.placeholder('float',[None,n_classes])
keep_rate = 0.75
DROPOUT=tf.placeholder(dtype=tf.float32)
print("sample")

  
def initialize_params():
    weights={}
    biases={}
    for layer in range(1,len(filters)):
        weights['Wconv'+str(layer)]=tf.Variable(tf.truncated_normal([patch[layer-1],patch[layer-1],filters[layer-1],filters[layer]],stddev=0.1,dtype=tf.float32))
        biases['bconv'+str(layer)]=tf.Variable(tf.constant(0.0,shape=[filters[layer]]))
    for layer in range(1,len(fc)):
        weights['Wfc'+str(layer)]=tf.Variable(tf.truncated_normal([fc[layer-1],fc[layer]],stddev=0.1,dtype=tf.float32))
        biases['bfc'+str(layer)]=tf.Variable(tf.constant(0.0,shape=[fc[layer]]))
    return weights,biases


def model(x,weights,biases):
    img=tf.reshape(x,[-1,image_size,image_size,1])
    Y=img
    for layer in range(1,len(filters)):
        Y=tf.nn.relu(tf.nn.conv2d(Y,weights['Wconv'+str(layer)],strides=[1,1,1,1],padding='SAME')+biases['bconv'+str(layer)])
        Y=tf.nn.max_pool(Y,ksize=[1,2,2,1],strides=[1,stride_size,stride_size,1],padding='SAME')
        
    Y_flattened=tf.reshape(Y,(-1,flattened_layer))
    for layer in range(1,len(fc)):
        if layer==len(fc)-1:
            Y_flattened=tf.matmul(Y_flattened,weights['Wfc'+str(layer)])+biases['bfc'+str(layer)]
            break
        Y_flattened=tf.nn.tanh(tf.matmul(Y_flattened,weights['Wfc'+str(layer)])+biases['bfc'+str(layer)])
        Y_flattened=tf.nn.dropout(Y_flattened, DROPOUT)
        
        
    output=Y_flattened
    return output

def train_neural_network(x):
    w,b=initialize_params()
    prediction = model(x,w,b)
    global num_train
    global num_test
    print("dimension of predicted value of NN: {}".format(prediction))# hypothesis
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    tf_pred = tf.argmax(prediction,1)
    optimizer = tf.train.RMSPropOptimizer(alpha,dr).minimize(cost)
    hm_epochs = 20
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss=0
        for i in range(int(num_train/batch_size)):
            epoch_x, epoch_y = helper.next_batch(Xtrain,labels,batch_size,indices)
            _, minibatchloss = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, DROPOUT: keep_rate})
            epoch_loss += minibatchloss/batch_size
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
    print('=========training successful===========')
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = 100*tf.reduce_mean(tf.cast(correct, 'float'))
    print('Training Accuracy : {}:'.format(accuracy.eval({x:Xtrain, y:labels,DROPOUT: keep_rate})))
    #scores=sess.run(correct,{x:Xtrain, y:labels,DROPOUT: keep_rate})
    print('computing predictions')
    test_pred=np.zeros(num_test)
    for i in range(int(num_test/batch_size)):
        test_pred[i*batch_size:(i+1)*batch_size]=tf_pred.eval(feed_dict={x:Xtest[i*batch_size:(i+1)*batch_size],DROPOUT: 1.0})
    sess.close()
    return test_pred

if __name__=="__main__":
    counter=0
    Xtrain,ytrain=helper.read_and_preprocess('train.csv')
    Xtest=helper.read_and_preprocess('test.csv')
    num_train=Xtrain.shape[0]
    num_test=Xtest.shape[0]
    imshow.showsamples(Xtrain,ytrain)
    indices=np.arange(num_train)
    labels = helper.dense_to_one_hot(ytrain)
    print(labels.shape)
    test_pred=train_neural_network(x)
    imshow.showpredictions(Xtest,test_pred)
    predictionfile = pd.DataFrame(data={'ImageId':(np.arange(num_test)+1), 'Label':test_pred.astype(np.uint8)})
    predictionfile.to_csv('predictions.csv',index=False)
    print("====success====")