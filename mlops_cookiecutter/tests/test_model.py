import torch
import os
import os
import sys
import pytest
from torch.nn.modules import module
  
# setting path
sys.path.append(os.getcwd() + "/mlops_cookiecutter")
from src.models.build_model import MyModel
from src.models.train_model import train


@pytest.mark.parametrize("batch_size,expected_dim", [(1,1), (10,10)])
def test_model(batch_size,expected_dim):
    data_train_path = os.getcwd() + "/mlops_cookiecutter/data/processed/train.pth"
    model_path = os.getcwd() + "/mlops_cookiecutter/models/checkpoint.pth"

    trainloader = torch.load(data_train_path)
    assert trainloader.dataset.tensors[0].shape==torch.Size([5000,28,28])
    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    assert model(trainloader.dataset.tensors[0][0:batch_size].view(batch_size,-1)).shape==torch.Size([expected_dim,10])

def test_training():
    try:
        train(1,False)
    except Exception as e:
        print(f"Training failed with: \n {e}")