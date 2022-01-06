import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, labels = batch
        output = self(images)
        loss = F.nll_loss(output.squeeze(),labels.long())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, labels = batch
        val_output = self(images)
        loss = F.nll_loss(val_output.squeeze(),labels.long())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        _, top_class = torch.exp(val_output).topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        acc = torch.mean(equals.type(torch.FloatTensor))
        self.log("accuracy",acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer