#coding=utf-8


import os
import cv2
import numpy as np
from math import sqrt
from collections import Counter

def file_save(org_path):
    temp = os.listdir(org_path)
    f_list = []
    res_list = []
    for i in temp:
        slices = i.split('.')
        result = slices[0]
        f_path = os.path.join(org_path,i)
        f_list.append(f_path)
        res_list.append(result)
    return f_list,res_list
        

def binary_img(path):
    img = cv2.imread(path,0)
    _,thres = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edge = cv2.Canny(thres,50,150)
    return edge

def data_save(f_list,res_list,ratio):
    nums = len(res_list)
    data = np.zeros((nums,28,28),dtype=np.float)    
    label = np.array(res_list)

    print('Loading...')

    for i in range(nums):
        data[i] = binary_img(f_list[i])

    print('Finished.')

    boundary = int(ratio*nums)
    idx = np.random.permutation(nums)
    train_idx,test_idx = idx[:boundary],idx[boundary:]
    train_data,train_label = data[train_idx],label[train_idx]
    test_data,test_label = data[test_idx],label[test_idx]

    return train_data,train_label,test_data,test_label

        
def kNN(test_unit,train_data,train_label,k):
    train_size = len(train_label)
    distance = [sqrt(np.sum((test_unit-train_data[i])**2)) for i in range(train_size)]
    sorted_idx = np.argsort(distance)
    top_k = [train_label[i] for i in sorted_idx[:k]]
    votes = Counter(top_k)
    result = votes.most_common()[0][0]

    return result



if __name__=='__main__':
    np.random.seed(666)
    R_nums = 0
    f_list,res_list = file_save(r'C:\Users\Administrator\Desktop\code\kNN\mnist_data\mnist_data')
    train_data,train_label,test_data,test_label = data_save(f_list,res_list,0.7)
    print('Counting...')
    for i in range(len(test_label)):
        result = kNN(test_data[i],train_data,train_label,3)
        if result == test_label[i]:
            R_nums += 1
        print('down')

    accuracy = R_nums/len(test_label)
    print('Finished.')
    print('Accuracy:',accuracy)
