import os

import torch


def test_dataset():
    data_train_path = os.getcwd() + "/mlops_cookiecutter/data/processed/train.pth"
    trainloader = torch.load(data_train_path)
    data_test_path = os.getcwd() + "/mlops_cookiecutter/data/processed/train.pth"
    testloader = torch.load(data_test_path)
    assert len(trainloader) == 157
    assert len(testloader) == 157, f"Testdataset has shape {len(testloader)} but should be 1000"



