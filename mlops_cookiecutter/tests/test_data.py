import torch
import os

data_train_path = os.getcwd() + "/data/processed/train.pth"
trainloader = torch.load(data_train_path)
data_test_path = os.getcwd() + "/data/processed/train.pth"
testloader = torch.load(data_train_path)

def test_dataset():
    data_train_path = os.getcwd() + "/data/processed/train.pth"
    trainloader = torch.load(data_train_path)
    data_test_path = os.getcwd() + "/data/processed/train.pth"
    testloader = torch.load(data_train_path)
    assert len(trainloader) == 157
    assert len(testloader) == 157, f"Testdataset has shape {len(testloader)} but should be 1000"



