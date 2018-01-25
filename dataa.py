# -*- coding: utf-8 -*-

import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib
import pandas as pd
from tqdm import tqdm


#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
metadata = pd.read_csv('C:/Users/Dell/Desktop/metadata.csv')
totaldata = (metadata.Label != 0).values.astype('bool')
print(totaldata.sum())#合格图片总数

metaseg = pd.read_csv('C:/Users/Dell/Desktop/metaseg.csv')
totalseg = (metaseg.Label != 0).values.astype('bool')
print(totalseg.sum())#合格图片总数

data_temp_list = []
for it, im in tqdm(enumerate(metadata[totaldata].Path.values),
                   total=totaldata.sum(), desc='Reading MRI to memory'):
        
    img = nib.load(im).get_data()
    data_temp_list.append(img)

data_temp_list = np.asarray(data_temp_list)
m = np.mean(data_temp_list)
s = np.std(data_temp_list)
    
print(m)
print(s)    
print(data_temp_list.shape)

##==================== GET NORMALIZE IMAGES
X_train_input = []
X_train_target = []

X_dev_input = []
X_dev_target = []

print("Validation")
for it, im in tqdm(enumerate(metadata[totaldata].Path.values),
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

for it, im in tqdm(enumerate(metaseg[totalseg].Path.values),
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
print(X_dev_input.shape)
print(X_dev_target.shape)

# with open(save_dir + 'dev_input.pickle', 'wb') as f:
#     pickle.dump(X_dev_input, f, protocol=4)
# with open(save_dir + 'dev_target.pickle', 'wb') as f:
#     pickle.dump(X_dev_target, f, protocol=4)
# del X_dev_input, X_dev_target


print("Train")
for it, im in tqdm(enumerate(metadata[totaldata].Path.values),
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

for it, im in tqdm(enumerate(metaseg[totalseg].Path.values),
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
print(X_train_input.shape)
print(X_train_target.shape)
#np.savez("testsave.npz",X_train_input,X_train_target,X_dev_input,X_dev_target)
#r=np.load("testsave.npz")
#rr=r["arr_0"]
#print(rr)

#save_dir='C:/Users/Dell/Desktop/ccc/'
#with open(save_dir + 'train_input.pickle', 'wb') as f:
#     pickle.dump(X_train_input, f, protocol=4)
#with open(save_dir + 'train_target.pickle', 'wb') as f:
#     pickle.dump(X_train_target, f, protocol=4)
#--------------------------------傲娇的分割线-----------------------------------


    
    
    
    
    
