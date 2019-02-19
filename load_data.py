import pickle
import numpy as np
from PIL import Image
import cv2

def load_CIFAR(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
##[b'labels':,  b'data':   ,b'filenames':   ,b'batch_label']

# test_data = load_CIFAR('G:/data/cifar-10-batches-py/test_batch')
# print(test_data)

#转成rgb图片的形式
def Getphoto(pixel):
    assert len(pixel) ==3072
    r = pixel[0:1024]; r = np.reshape(r,[32,32,1])
    g = pixel[1024:2048];g = np.reshape(g,[32,32,1])
    b = pixel[2048:3072];b = np.reshape(b,[32,32,1])

    photo = np.concatenate([r,g,b],-1)

    return photo

# a = test_data[b'data']
# a = np.array(a)
# print(a.shape)
#
# b = Getphoto(a[0])
# print(b.shape)

##Image显示
# im = Image.fromarray(b).convert('RGB')
# im.show()
##opencv 显示
# img = cv2.imread('G:/picture/demo1.jpg')
# img = Image.open('G:/picture/demo1.jpg')
# cv2.imshow('hahah',img)
# print(type(img))


# cv2.imshow('test',b)
# cv2.waitKey(0)
# cv2.destroyWindow('test')
##获取训练数据集
def GetTrainDataByLable(label):
    batch_label = []
    labels = []
    data = []
    filenames = []

    for i in range(1,6):
        batch_label.append(load_CIFAR('G:/data/cifar-10-batches-py/data_batch_%d'%i)[b'batch_label'])
        labels += load_CIFAR('G:/data/cifar-10-batches-py/data_batch_%d'%i)[b'labels']
        data.append(load_CIFAR('G:/data/cifar-10-batches-py/data_batch_%d'%i)[b'data'])
        filenames += load_CIFAR('G:/data/cifar-10-batches-py/data_batch_%d'%i)[b'filenames']
    # print(len(data))
    data = np.concatenate(data,0)#拼接 按横向拼接
    # print(len(data))
    # label = str.encode(label)
    # print(label)
    if label == b'data':
        print()
        array = np.ndarray([len(data),32,32,3],dtype=np.float32)
        print(type(array))
        for i in range(len(data)):
            array[i] = Getphoto(data[i])
        return array
        pass
    elif label == b'labels':
        array_label = np.zeros([len(labels),10],dtype=np.float32)
        for i in range(len(labels)):
            array_label[i][labels[i]] = 1.0
        return array_label
        # return np.array(labels)
        pass
    elif label == b'batch_label':
        return batch_label
    elif label == b'filenames':
        return filenames
    else:
        raise NameError

# train_data = GetTrainDataByLable(b'data')
# print(train_data.shape)
#
lable_data = GetTrainDataByLable(b'labels')
print(lable_data)
print(lable_data.shape)

##获取测试数据集

def GetTestData(label):
    batch_label = []
    filenames = []
    batch_label.append(load_CIFAR('G:/data/cifar-10-batches-py/test_batch')[b'batch_label'])
    filenames += load_CIFAR('G:/data/cifar-10-batches-py/test_batch')[b'filenames']
    data = load_CIFAR('G:/data/cifar-10-batches-py/test_batch')[b'data']
    labels = load_CIFAR('G:/data/cifar-10-batches-py/test_batch')[b'labels']

    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3],dtype=np.float32)
        for i in range(len(data)):
            array[i] = Getphoto(data[i])
        return array
        pass
    elif label == b'label':
        return np.array(labels)
        pass
    elif label == b'batch_label':
        return batch_label
        pass
    elif label == b'filenames':
        return filenames
    else:
        raise NameError


# test_data = GetTestData(b'data')
# print(test_data.shape)



