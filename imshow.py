#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:04:16 2018

@author: kulka193
"""
import numpy as np
import matplotlib.pyplot as plt
image_size=28
num_images_shown=5

def showsamples(Xtrain,ytrain):
    num_train=Xtrain.shape[0]
    samples=np.random.randint(num_train,size=5)
    for k in samples:
        plt.title('Image of index {}'.format(k+1))
        plt.imshow(Xtrain[k,:].reshape(image_size,image_size),cmap='gray')
        plt.show()
        print('labels for indices {} : {}'.format((k+1),ytrain[k]))

def showpredictions(Xtest,pred_images):
    num_test=Xtest.shape[0]
    samples=np.random.randint(num_test,size=num_images_shown)
    for k in samples:
        plt.title('Image of index {}'.format(k+1))
        plt.imshow(Xtest[k,:].reshape(image_size,image_size),cmap='gray')
        plt.show()
        print('labels for indices {} : {}'.format((k+1),pred_images[k]))

        
        
        
        
