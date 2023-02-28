# dataset used: https://www.kaggle.com/datasets/biancaferreira/african-wildlife

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tqdm import tqdm

from utils.dataset import Dataset
from utils.visualize import inverse_transform
from utils.extra import collate_fn


EPOCHS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset stuff
dataset = Dataset("data", (480, 480))
train_dataset, val_dataset = random_split(dataset, [1300, 201])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# model stuff
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
for param in model.backbone.parameters():
    param.requires_grad = False
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

model.to(device)

# optim stuff
optimizer = Adam(model.roi_heads.parameters(), lr=0.0001)


for epoch in range(EPOCHS):
    model.train()
    # train
    for (images, annotations) in tqdm(train_dataloader, total=int(1300/32)+1):
        optimizer.zero_grad()

        loss_dict = model(images, annotations)

        del images
        del annotations

        torch.cuda.empty_cache()

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        
        print("Train loss: " + str(losses))

        optimizer.step()
    
    with torch.no_grad():
        # validation
        for i, (images, annotations) in enumerate(tqdm(val_dataloader, total=int(201/32)+1)):
            loss_dict = model(images, annotations)

            if i % 6 == 0 and i != 0:
                model.eval()
                output = model(images[0].unsqueeze(0))[0]
                print(output["labels"])
                print(annotations[0]["labels"])
                inverse_transform(images[0], output, "outputs/" + str(epoch) + ".jpg")

            del images
            del annotations

            torch.cuda.empty_cache()

            losses = sum(loss for loss in loss_dict.values())
            print("Val loss: " + str(losses))