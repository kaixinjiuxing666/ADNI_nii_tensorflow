# -*- coding: utf-8 -*-

import numpy as np
import os, random, gc, pickle
import nibabel as nib
import pandas as pd
from tqdm import tqdm



#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD==============
#.csv是存放表格数据的格式，可用excel直接打开并编辑
metadata = pd.read_csv('/home/zhengyz/oooooo/new/data25622.csv')
totaldata = (metadata.Label != 0).values.astype('bool')
print(totaldata.sum())#合格图片总数
 
metaseg = pd.read_csv('/home/zhengyz/oooooo/new/seg25622.csv')
totalseg = (metaseg.Label != 0).values.astype('bool')
print(totalseg.sum())#合格图片总数

#val:192*192*160-74组:15
#tra:192*192*160-74组:59   cup_feature_guard
#val:256*256*166-36组:6
#tra:256*256*166-36组:30
#val:256*256*180-22组:4
#tra:256*256*180-22组:18


data = []
for it, im in tqdm(enumerate(metadata[totaldata].Path.values),
                   total=totaldata.sum(), desc='Reading MRI to memory'):
        
    img = nib.load(im).get_data()
    data.append(img)
    
data = np.asarray(data)
m = np.mean(data)
s = np.std(data)
    
print(m)
print(s)    
print(data.shape)

##==================== GET NORMALIZE IMAGES====================================
X_train_input = []
X_train_target = []

X_dev_input = []
X_dev_target = []

print("Validation")
for it, im in tqdm(enumerate(metadata[:4].Path.values),
                   total=totaldata.sum(),desc='Reading MRI to memory'):        
    all_3d_data = []
    img = nib.load(im).get_data()
    img = (img - m) / s
    img = img.astype(np.float32)
    all_3d_data.append(img)
       
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.transpose(all_3d_data[0][:,:,j], (1, 0, 2))#.tolist()   
        combined_array.astype(np.float32)
        X_dev_input.append(combined_array)
    
    del all_3d_data
    gc.collect()

for it, im in tqdm(enumerate(metaseg[:4].Path.values),
                   total=totalseg.sum(),desc='Reading LABEL to memory'):
    all_3d_seg = []
    seg_img = nib.load(im).get_data()
    seg_img = seg_img.astype(np.float32)
    all_3d_seg.append(seg_img)
    
    for j in range(all_3d_seg[0].shape[2]):
        combined_array = np.transpose(all_3d_seg[0][:,:,j], (1, 0, 2))#.tolist()   
        combined_array.astype(int)
        X_dev_target.append(combined_array)

    del all_3d_seg
    gc.collect()

X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
X_dev_target = np.asarray(X_dev_target)#, dtype=np.float32)
X_dev_target = np.minimum(X_dev_target,1)
print(X_dev_input.shape)
print(X_dev_target.shape)


print("Train")
for it, im in tqdm(enumerate(metadata[4:].Path.values),
                   total=totaldata.sum(),desc='Reading MRI to memory'):        
    all_3d_data = []
    img = nib.load(im).get_data()
    img = (img - m) / s
    img = img.astype(np.float32)
    all_3d_data.append(img)
       
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.transpose(all_3d_data[0][:,:,j], (1, 0, 2))#.tolist()   
        combined_array.astype(np.float32)
        X_train_input.append(combined_array)
    del all_3d_data
    gc.collect()

for it, im in tqdm(enumerate(metaseg[4:].Path.values),
                   total=totalseg.sum(),desc='Reading LABEL to memory'):
    all_3d_seg = []
    seg_img = nib.load(im).get_data()
    seg_img = seg_img.astype(np.float32)
    all_3d_seg.append(seg_img)
    
    for j in range(all_3d_seg[0].shape[2]):
        combined_array = np.transpose(all_3d_seg[0][:,:,j], (1, 0, 2))#.tolist()   
        combined_array.astype(int)
        X_train_target.append(combined_array)

    del all_3d_seg
    gc.collect()

X_train_input = np.asarray(X_train_input, dtype=np.float32)
X_train_target = np.asarray(X_train_target)#, dtype=np.float32)
X_train_target = np.minimum(X_train_target,1)
print(X_train_input.shape)
print(X_train_target.shape)

np.savez("256save22.npz",X_train_input,X_train_target,X_dev_input,X_dev_target)
#r=np.load("testsave.npz")
#rr=r["arr_0"]
#print(rr)


#save_dir='C:/Users/Dell/Desktop/ccc/'
#with open(save_dir + 'train_input.pickle', 'wb') as f:
#     pickle.dump(X_train_input, f, protocol=4)
#with open(save_dir + 'train_target.pickle', 'wb') as f:
#     pickle.dump(X_train_target, f, protocol=4)


    
    
    
    
    
