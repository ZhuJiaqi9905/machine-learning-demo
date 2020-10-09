import numpy as np 
import operator
def create_dataset():
    '''创造数据集'''
    group = np.array([[1.0, 1.1],[1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    delta_v = np.tile(inX, (dataset_size, 1)) - dataset
    delta_v = delta_v**2
    delta_v = delta_v.sum()
    distance = delta_v**0.5
    sorted_distance = distance.argsort()
    classCount = {}
    
