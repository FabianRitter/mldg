import numpy as np
import pickle
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def write(fpath, text):
    with open(fpath, 'a+') as f:
        f.write(text + '\n')