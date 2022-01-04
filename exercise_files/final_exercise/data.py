from torch import Tensor, randn
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
def mnist():

    # Download and load the training data
    path_1 = str(Path.cwd().parent.parent) + "/dtu_mlops/data/corruptmnist/train_0.npz"
    trainset = np.load(path_1)
    images = Tensor(trainset["images"]) # transform to torch tensor
    labels = Tensor(trainset["labels"])
    my_dataset = TensorDataset(images,labels)
    train = DataLoader(my_dataset, batch_size=32, shuffle=True)
    path_2 = str(Path.cwd().parent.parent) + "/dtu_mlops/data/corruptmnist/test.npz"
    testset = np.load(path_2)
    images = Tensor(testset["images"]) # transform to torch tensor
    labels = Tensor(testset["labels"])
    my_dataset = TensorDataset(images,labels)
    test = DataLoader(my_dataset, batch_size=32, shuffle=True)    
    return train, test
    
if __name__ == '__main__':
    mnist()