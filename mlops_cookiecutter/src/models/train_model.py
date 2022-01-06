import torch
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from build_model import MyModel



def train():
    log = logging.getLogger(__name__)
    log.info("Training day and night")
    
    model_path = os.getcwd() + "/models/checkpoint.pth"
    data_train_path = os.getcwd() + "/data/processed/train.pth"
    data_test_path = os.getcwd() + "/data/processed/test.pth"

    # add any additional argument that you want

    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    trainloader = torch.load(data_train_path)
    valloader = torch.load(data_test_path)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(max_epochs=20,logger=logger)
    trainer.fit(model,trainloader)
    trainer.validate(model,valloader)

    
    save_pth = os.getcwd() + '/models/checkpoint_new.pth'
    torch.save(model.state_dict(),save_pth)
    log.info(f"Model saved to: {save_pth}")

if __name__ == '__main__':
    train()