import os
import image
import numpy as np
from PIL import Image
from math import sqrt
from collections import Counter

def binaryzation(data):
    row = data.shape[0]
    col = data.shape[1]
    matrix = np.zeros((row,col),dtype='int')
    for i in range(row):
        for j in range(col):
            if data[i][j] > 127:
                matrix[i][j] = 1

    return matrix

def load_data(path, ratio):
    files = os.listdir(path)
    file_num = len(files)

    img_mat = np.empty((file_num,28,28),dtype = 'float')
    data = np.empty((file_num,28,28),dtype = 'float')
    label = np.empty((file_num), dtype = 'int')

    print('Loading data...')

    for i in range(file_num):
        print(i+1,'/',file_num,'\r')
        file_name = files[i]
        file_path = os.path.join(path,file_name)
        img_mat[i] = Image.open(file_path)
        data[i] = binaryzation(img_mat[i])
        #print(data[i])
        label[i] = int(file_name.split('.')[0])
        #print(label[i])
    print('Finished')

    boundary = int(ratio*file_num)
    idx = np.random.permutation(file_num)
    train_idx,test_idx = idx[:boundary],idx[boundary:]
    train_data,test_data = data[train_idx],data[test_idx]
    train_label,test_label = label[train_idx],label[test_idx]

    
    return train_data,train_label,test_data,test_label

def kNN(test_unit,train_data,train_label,k):
    train_size = len(train_label)
    distance = [sqrt(np.sum((test_unit-train_data[i])**2)) for i in range(train_size)]
    sorted_idx = np.argsort(distance)

    Top_k = [train_label[i] for i in sorted_idx[:k]]
    votes = Counter(Top_k)
    result = votes.most_common()[0][0]

    return result

if __name__ == '__main__':
    np.random.seed(666)

    train_data,train_label,test_data,test_label = load_data('mnist_data',0.7)

    test_num = len(test_label)
    Right_num = 0

    print('Testing...')
    
    for i in range(test_num):
        print(i+1,'/',test_num,'\r')
        result = kNN(test_data[i],train_data,train_label,4)
        if result == test_label[i]:
            Right_num += 1

    accuracy = Right_num/test_num
    print('Accuracy:',accuracy)
