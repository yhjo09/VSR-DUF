## -*- coding: utf-8 -*-
import tensorflow as tf

from utils import BatchNorm, Conv3D

stp = [[0,0], [1,1], [1,1], [1,1], [0,0]]
sp = [[0,0], [0,0], [1,1], [1,1], [0,0]]

def FR_16L(x, is_train):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,3,64], [1,1,1,1,1], 'VALID', name='conv1') 

    F = 64
    G = 32
    for r in range(3):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x, t], 4)
        F += G
    for r in range(3,6):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x[:,1:-1], t], 4)
        F += G

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,256,256], [1,1,1,1,1], 'VALID', name='conv2')
    x = tf.nn.relu(x)
    
    r = Conv3D(x, [1,1,1,256,256], [1,1,1,1,1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1,1,1,256,3*16], [1,1,1,1,1], 'VALID', name='rconv2')  
    
    f = Conv3D(x, [1,1,1,256,512], [1,1,1,1,1], 'VALID', name='fconv1') 
    f = tf.nn.relu(f)
    f = Conv3D(f, [1,1,1,512,1*5*5*16], [1,1,1,1,1], 'VALID', name='fconv2')    
    
    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
    f = tf.nn.softmax(f, dim=4)

    return f, r

def FR_28L(x, is_train):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,3,64], [1,1,1,1,1], 'VALID', name='conv1')

    F = 64
    G = 16
    for r in range(9):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x, t], 4)
        F += G
    for r in range(9,12):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x[:,1:-1], t], 4)
        F += G
    
    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,256,256], [1,1,1,1,1], 'VALID', name='conv2')

    x = tf.nn.relu(x)
    
    r = Conv3D(x, [1,1,1,256,256], [1,1,1,1,1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1,1,1,256,3*16], [1,1,1,1,1], 'VALID', name='rconv2')  
    
    f = Conv3D(x, [1,1,1,256,512], [1,1,1,1,1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1,1,1,512,1*5*5*16], [1,1,1,1,1], 'VALID', name='fconv2')    
    
    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
    f = tf.nn.softmax(f, dim=4)

    return f, r

def FR_52L(x, is_train):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,3,64], [1,1,1,1,1], 'VALID', name='conv1')

    F = 64
    G = 16
    for r in range(0,21):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x, t], 4)
        F += G
    for r in range(21,24):
        t = BatchNorm(x, is_train, name='Rbn'+str(r+1)+'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1,1,1,F,F], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'a') 
        
        t = BatchNorm(t, is_train, name='Rbn'+str(r+1)+'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3,3,3,F,G], [1,1,1,1,1], 'VALID', name='Rconv'+str(r+1)+'b') 
        
        x = tf.concat([x[:,1:-1], t], 4)
        F += G

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1,3,3,448,256], [1,1,1,1,1], 'VALID', name='conv2')

    x = tf.nn.relu(x)
    
    r = Conv3D(x, [1,1,1,256,256], [1,1,1,1,1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1,1,1,256,3*16], [1,1,1,1,1], 'VALID', name='rconv2')  
    
    f = Conv3D(x, [1,1,1,256,512], [1,1,1,1,1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1,1,1,512,1*5*5*16], [1,1,1,1,1], 'VALID', name='fconv2')    
    
    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
    f = tf.nn.softmax(f, dim=4)

    return f, r