## -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import glob
import scipy
import argparse
import os
from PIL import Image

from utils import LoadImage, DownSample, AVG_PSNR, depth_to_space_3D, DynFilter3D, LoadParams
from nets import FR_16L, FR_28L, FR_52L

parser = argparse.ArgumentParser()
parser.add_argument('R', metavar='R', type=int, help='Upscaling factor: One of 2, 3, 4')
parser.add_argument('L', metavar='L', type=int, help='Network depth: One of 16, 28, 52')
parser.add_argument('T', metavar='T', help='Input type: L(Low-resolution) or G(Ground-truth)')
args = parser.parse_args()

# Size of input temporal radius
T_in = 7
# Selecting filters and residual generating network
if args.L == 16:
    FR = FR_16L
elif args.L == 28:
    FR = FR_28L
elif args.L == 52:
    FR = FR_52L
else:
    print('Invalid network depth: {} (Must be one of 16, 28, 52)'.format(args.L))
    exit(1)

if not(args.T == 'L' or args.T =='G'):
    print('Invalid input type: {} (Must be L(Low-resolution) or G(Ground-truth))'.format(args.T))
    exit(1)


# Gaussian kernel for downsampling
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

# Upscaling factor
R = args.R
if R == 2:
    h = gkern(13, 0.8)  # 13 and 0.8 for x2
elif R == 3:
    h = gkern(13, 1.2)  # 13 and 1.2 for x3
elif R == 4:
    h = gkern(13, 1.6)  # 13 and 1.6 for x4
else:
    print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(args.R))
    exit(1)
h = h[:,:,np.newaxis,np.newaxis].astype(np.float32)


def G(x, is_train):  
    # shape of x: [B,T_in,H,W,C]

    # Generate filters and residual
    # Fx: [B,1,H,W,1*5*5,R*R]
    # Rx: [B,1,H,W,3*R*R]
    Fx, Rx = FR(x, is_train, uf=R) 

    x_c = []
    for c in range(3):
        t = DynFilter3D(x[:,T_in//2:T_in//2+1,:,:,c], Fx[:,0,:,:,:,:], [1,5,5]) # [B,H,W,R*R]
        t = tf.depth_to_space(t, R) # [B,H*R,W*R,1]
        x_c += [t]
    x = tf.concat(x_c, axis=3)   # [B,H*R,W*R,3]
    x = tf.expand_dims(x, axis=1)

    Rx = depth_to_space_3D(Rx, R)   # [B,1,H*R,W*R,3]
    x += Rx
    
    return x

# Network
H = tf.placeholder(tf.float32, shape=[None, T_in, None, None, 3])
L_ = DownSample(H, h, R)
L = L_[:,:,2:-2,2:-2,:]    # To minimize boundary artifact

is_train = tf.placeholder(tf.bool, shape=[]) # Phase ,scalar

with tf.variable_scope('G') as scope:
    GH = G(L, is_train)

params_G = [v for v in tf.global_variables() if v.name.startswith('G/')]

# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    # Load parameters
    LoadParams(sess, [params_G], in_file='params_{}L_x{}.h5'.format(args.L, R))
    
    if args.T == 'G':
        # Test using GT videos
        avg_psnrs = []
        dir_inputs = glob.glob('./inputs/G/*')
        for v in dir_inputs:
            scene_name = v.split('/')[-1]
            os.mkdir('./results/{}L/G/{}/'.format(args.L, scene_name))
            
            dir_frames = glob.glob(v + '/*.png')
            dir_frames.sort()

            frames = []
            for f in dir_frames:
                frames.append(LoadImage(f))
            frames = np.asarray(frames)
            frames_padded = np.lib.pad(frames, pad_width=((T_in//2,T_in//2),(0,0),(0,0),(0,0)), mode='constant')
            if R == 2:
                frames_padded = np.lib.pad(frames_padded, pad_width=((0,0),(2*R,2*R),(2*R,2*R),(0,0)), mode='reflect')
            elif R == 3:
                H_h, H_w = frames.shape[1:3]
                pad_h = 3 - (H_h % 3)
                pad_w = 3 - (H_w % 3)
                frames_padded = np.lib.pad(frames_padded, pad_width=((0,0),(2*R,2*R+pad_h),(2*R,2*R+pad_w),(0,0)), mode='reflect')
            elif R == 4:
                frames_padded = np.lib.pad(frames_padded, pad_width=((0,0),(2*R,2*R),(2*R,2*R),(0,0)), mode='reflect')
            
            out_Hs = []
            for i in range(frames.shape[0]):
                print('Scene {}: Frame {}/{} processing'.format(scene_name, i+1, frames.shape[0]))
                in_H = frames_padded[i:i+T_in]  # select T_in frames
                in_H = in_H[np.newaxis,:,:,:,:]
                
                out_H = sess.run(GH, feed_dict={H: in_H, is_train: False})
                out_H = np.clip(out_H, 0, 1)
                if R == 3:
                    if pad_h > 0:
                        out_H = out_H[:,:,:-pad_h,:,:]
                    if pad_w > 0:
                        out_H = out_H[:,:,:,:-pad_w,:]
                
                Image.fromarray(np.around(out_H[0,0]*255).astype(np.uint8)).save('./results/{}L/G/{}/Frame{:03d}.png'.format(args.L, scene_name, i+1))

                out_Hs.append(out_H[0, 0])
            out_Hs = np.asarray(out_Hs)
                
            avg_psnr = AVG_PSNR(((frames)*255).astype(np.uint8)/255.0, ((out_Hs)*255).astype(np.uint8)/255.0, vmin=0, vmax=1, t_border=2, sp_border=8)
            avg_psnrs.append(avg_psnr)
            print('Scene {}: PSNR {}'.format(scene_name, avg_psnr))

    elif args.T == 'L':
        # Test using Low-resolution videos
        dir_inputs = glob.glob('./inputs/L/*')
        for v in dir_inputs:
            scene_name = v.split('/')[-1]
            os.mkdir('./results/{}L/L/{}/'.format(args.L, scene_name))
            
            dir_frames = glob.glob(v + '/*.png')
            dir_frames.sort()

            frames = []
            for f in dir_frames:
                frames.append(LoadImage(f))
            frames = np.asarray(frames)
            frames_padded = np.lib.pad(frames, pad_width=((T_in//2,T_in//2),(0,0),(0,0),(0,0)), mode='constant')
            
            for i in range(frames.shape[0]):
                print('Scene {}: Frame {}/{} processing'.format(scene_name, i+1, frames.shape[0]))
                in_L = frames_padded[i:i+T_in]  # select T_in frames
                in_L = in_L[np.newaxis,:,:,:,:]
                
                out_H = sess.run(GH, feed_dict={L: in_L, is_train: False})
                out_H = np.clip(out_H, 0, 1)

                Image.fromarray(np.around(out_H[0,0]*255).astype(np.uint8)).save('./results/{}L/L/{}/Frame{:03d}.png'.format(args.L, scene_name, i+1))
                
            
