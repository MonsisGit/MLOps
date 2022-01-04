import argparse
import sys
import torch

from torch import optim
from torch import nn

from matplotlib import pyplot as plt

from build_model import MyModel



def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--epochs',type=int,nargs=1,default=30)
    parser.add_argument('--model_path',default="models/checkpoint.pth")
    parser.add_argument('--data_train_path',default="data/processed/train.pth")
    parser.add_argument('--data_test_path',default="data/processed/test.pth")

    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    model = MyModel()
    model.load_state_dict(torch.load(args.model_path))
    trainloader = torch.load(args.data_train_path)
    testloader = torch.load(args.data_test_path)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    epochs = args.epochs

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

        print(f"Training loss: {running_loss/len(trainloader)}")
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
            print(f'Accuracy: {acc*100:.2f}%')
            test_loss.append(running_loss/len(testloader))
            accuracies.append(acc)

    torch.save(model.state_dict(),'models/checkpoint_new.pth')

    plt.figure(figsize=(16,8))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.plot(accuracies)
    plt.legend(["train loss","test loss","accuracies"])
    plt.savefig("reports/figures/loss.png")

if __name__ == '__main__':
    train()