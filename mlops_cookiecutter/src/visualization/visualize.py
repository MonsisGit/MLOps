import argparse
import sys

import torch

from matplotlib import pyplot as plt
from build_model import MyModel


def predict():

    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--model_path', default="models/checkpoint.pth")
    parser.add_argument('--folder_path',default="data/processed/test_1.pth")

    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    testloader = torch.load(args.folder_path)
    
    model = MyModel()
    model.load_state_dict(torch.load(args.model_path))


    with torch.no_grad():
        cnt = 0
        for images, labels in testloader:  
            for idx,img in enumerate(images):    

                img = img.view(1,28,28) 
                log_ps = model(img)

                _, top_class = torch.exp(log_ps).topk(1, dim=1)
                plt.figure(figsize=(10,5))
                plt.title(f"Label: {labels[idx]}, Prediction: {top_class.item()}")
                plt.imshow([img[0]])
                plt.savefig("reports/visualisations/img_" + str(idx) + ".png")

                if labels[idx]==top_class:
                    cnt+=1
            break
        print(f'Accuracy: {cnt/idx*100:.2f}%')

if __name__ == '__main__':
    predict()