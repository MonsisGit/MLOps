import numpy as np
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
from torch import Tensor, save
from torch.utils.data import DataLoader, TensorDataset


def mnist():

    # Download and load the training data
    path_1 = "data/raw/train_0.npz"
    trainset = np.load(path_1)
    images = Tensor(trainset["images"]) # transform to torch tensor
    labels = Tensor(trainset["labels"])
    my_dataset = TensorDataset(images,labels)
    train = DataLoader(my_dataset, batch_size=32, shuffle=True,num_workers=4)
    path_2 = "data/raw/test.npz"
    testset = np.load(path_2)
    images = Tensor(testset["images"]) # transform to torch tensor
    labels = Tensor(testset["labels"])
    my_dataset = TensorDataset(images,labels)
    test = DataLoader(my_dataset, batch_size=32, shuffle=False,num_workers=4)    
    save(train,"data/processed/train.pth")
    save(test,"data/processed/test.pth")
    print("Saved to data/processed/")


if __name__ == '__main__':
    mnist()
