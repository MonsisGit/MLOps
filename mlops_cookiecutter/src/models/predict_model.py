import argparse
import logging
import sys

import hydra
import torch
from build_model import MyModel


@hydra.main(config_name="config_train",config_path="../config/")
def predict(cfg):

    log = logging.getLogger(__name__)

    log.info("Training day and night")
    
    model_path = cfg.hyperparameters.model_path
    data_test_path = cfg.hyperparameters.data_test_path

    testloader = torch.load(data_test_path)

    model = MyModel()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        cnt = 0
        for images, labels in testloader:  
            for idx,img in enumerate(images):    

                img = img.view(1,28,28) 
                log_ps = model(img)

                _, top_class = torch.exp(log_ps).topk(1, dim=1)
                print(f"Label: {labels[idx]}, Prediction: {top_class.item()}")
                if labels[idx]==top_class:
                    cnt+=1
            break
        print(f'Accuracy: {cnt/idx*100:.2f}%')

if __name__ == '__main__':
    predict()