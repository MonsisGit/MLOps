import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel
from torch import optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.003)
        epochs = 50
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                # Flatten MNIST images into a 784 long vector
                optimizer.zero_grad()
                # TODO: Training pass
                output = model(images)
                loss = criterion(output.squeeze(),labels.long())
                loss.backward()        
                running_loss += loss.item()
                optimizer.step()
               # import pdb
               # pdb.set_trace()
            else:
                print(f"Training loss: {running_loss/len(train_set)}")
                
        torch.save(model.state_dict(),'checkpoint.pth')
        
    
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, testloader = mnist()
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()        
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
            print(f'Accuracy: {acc*100:.2f}%')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    