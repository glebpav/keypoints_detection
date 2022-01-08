import torch
import config
import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
from model import FaceKeypointResNet50
import time

from printing_utils import *
from settings import CONFIDENCE_LIMIT

model = FaceKeypointResNet50(pretrained=True, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
# TODO PATH TO MODEL
checkpoint = torch.load('model_dataset9_80.pth', map_location='cpu')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# TODO PATH DATASET
test_data_file = open("data_minus.csv", 'r')


for current_str in test_data_file:

    current_str = current_str.replace('"', '')
    picture_name = current_str[0:current_str.find(";")]
    current_str = current_str[current_str.find(";") + 1:]
    k = current_str.split(";")[0:-1]

    dataset_points = []
    predicted_points = []
    x, y, confidence = 0, 0, 0

    for idx, item in enumerate(k):
        item = item.replace('"', '')
        if idx % 3 == 0:
            confidence = int(item)
        if idx % 3 == 1:
            x = float(item)
        if idx % 3 == 2:
            y = float(item)

            dataset_points.append(Animals_points(x, y, confidence, animals_points_classes[int(idx / 3)]))
            # print(Animals_points(x, y, confidence, animals_points_classes[int(idx / 3)]))
    # picture_path = "F:/Dataset/data_for_testing/" + name

    # TODO PATH TO IMAGES
    picture_path = 'D:\shool staff\projects\Animal Posing (21-22)\JPEGImages\\' + \
                   picture_name.split('_')[0] + '\\' + picture_name
    # print(picture_path)

    if not os.path.exists(picture_path):
        continue

    cap = cv2.VideoCapture(picture_path)

    # capture each frame of the video
    ret, frame = cap.read()

    with torch.no_grad():
        image = frame
        h, w = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)
        outputs = model(image)

        for i in outputs:
            x, y, confidence = 0, 0, 0
            for idx, j in enumerate(i):
                if idx % 3 == 0:
                    confidence = float(str(j)[7:len(str(j)) - 1])
                if idx % 3 == 1:
                    x = float(str(j)[7:len(str(j)) - 1])
                if idx % 3 == 2:
                    y = float(str(j)[7:len(str(j)) - 1])
                    predicted_points.append(Animals_points(x, y, confidence, animals_points_classes[int(idx / 3)]))

    outputs = outputs.cpu().detach().numpy()

    # predicted
    frame = add_points(frame, predicted_points, w, h, (255, 0, 0), print_txt=True)
    # frame = add_connections(frame, predicted_points, w, h, True)

    # real (from dataset)
    frame = add_points(frame, dataset_points, print_txt=True)
    # frame = add_connections(frame, dataset_points)

    plt.imshow(frame)
    plt.show()
