from utils.datasets_distance import *
from roipool import *
from model_dist import *

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import cv2
import time

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg16 = torchvision.models.vgg16(pretrained=True).to(device)

    feature_extractor = vgg16.features
    feature_extractor.eval()
    
    for param in feature_extractor.parameters():
        param.requires_grad = False

    roipool = ROIPool((2, 2)).to(device)
    roipool.eval()

    model_distance = Dist().to(device)
    model_distance.load_state_dict(torch.load('checkpoints/distance.pth'))

    dataset = ListDataset('data/train.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model_distance.parameters(), lr=0.001) 

    for epoch in range(2000):
        model_distance.train()
        total_loss = 0
        for batch_i, (img_path, imgs, targets, distance) in enumerate(dataloader):
            print(targets)
            print(distance)
            
            # print(feature_extractor[0].bias)
            imgs = imgs.to(device)
            bboxes = targets.to(device)
            distance = distance.float().to(device)

            optimizer.zero_grad()

            feature_map = feature_extractor(imgs)
            
            for i in range(len(bboxes[0])):
                roi = roipool(feature_map, bboxes[0][i])
                output = model_distance(roi)
                target = distance[0][i]
                target = target.unsqueeze(0)
                
                loss = loss_fn(output, target)
                
                loss.backward()
            optimizer.step()


        torch.save(model_distance.state_dict(), f'checkpoints/distance.pth')

        