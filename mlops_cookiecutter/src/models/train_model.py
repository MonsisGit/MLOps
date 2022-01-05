import argparse
import sys
import torch

from torch import optim
from torch import nn
import hydra
from hydra.utils import get_original_cwd
import logging
from matplotlib import pyplot as plt

from build_model import MyModel



@hydra.main(config_name="config_train",config_path="../config/")
def train(cfg):
    log = logging.getLogger(__name__)
    log.info("Training day and night")
    
    epochs = cfg.hyperparameters.epochs
    model_path = get_original_cwd() + cfg.hyperparameters.model_path
    data_train_path = get_original_cwd() + cfg.hyperparameters.data_train_path
    data_test_path = get_original_cwd() + cfg.hyperparameters.data_test_path
    lr = cfg.hyperparameters.lr

    # add any additional argument that you want

    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    trainloader = torch.load(data_train_path)
    testloader = torch.load(data_test_path)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_loss, test_loss, accuracies = [],[],[]
    for _ in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            optimizer.zero_grad()
            # TODO: Training pass
            output = model(images)
            loss = criterion(output.squeeze(),labels.long())
            loss.backward()        
            running_loss += loss.item()
            optimizer.step()

        log.info(f"Training loss: {running_loss/len(trainloader)}")
        train_loss.append(running_loss/len(trainloader))

        with torch.no_grad():
            cnt = []
            running_loss = 0
            for images, labels in testloader:     
                  
                log_ps = model(images)
                loss = criterion(log_ps, labels.long())
                running_loss += loss.item()
                _, top_class = torch.exp(log_ps).topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                cnt.append(torch.mean(equals.type(torch.FloatTensor)))  
            acc = torch.sum(torch.tensor(cnt))/len(cnt)
            log.info(f'Accuracy: {acc*100:.2f}%')
            test_loss.append(running_loss/len(testloader))
            accuracies.append(acc)

    save_pth = get_original_cwd() + '/models/checkpoint_new.pth'
    torch.save(model.state_dict(),save_pth)
    log.info(f"Model saved to: {save_pth}")

    plt.figure(figsize=(16,8))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.plot(accuracies)
    plt.legend(["train loss","test loss","accuracies"])
    plt.savefig(get_original_cwd() + "/reports/figures/loss.png")

if __name__ == '__main__':
    train()