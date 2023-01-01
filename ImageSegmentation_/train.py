import torch 
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import torch.optim as optim
from Model import UNET
from tqdm import tqdm
from utils import(
    load_checkpoint, 
    save_checkpoint,
    get_loaders, 
    check_accuracy,
    save_prediction_as_imgs,
)

#HYPERPARAMETERS
LEARNING_RATE = 0.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 3
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_HEIGHT = 160
IMG_WEIDTH = 240
PIN_MEMORY = True
LOAD_MEMORY = True
TRAIN_IMAGE_DIR = '.\Data\train'
TRAIN_MASK_DIR = '.\Data\train_masks'
VAL_IMAGE_DIR = '\Data\val'
VAL_MASK_DIR = '.\Data\val_masks'

def train_fn(loader, model, optimizer, loss_fn, scalar):
    loop = tqdm(loader)

    for batch_indx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, targets)

        #backward
        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        #udate tqdm
        loop.set_postfix(loss = loss.itme())

def main():
    train_transforms = A.compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WEIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WEIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std= [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = UNET(inchannels=3, outchannels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms
    )

    scalar = torch.cuda.amp.grad_scaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scalar)

        #save model
        check_point = {
            "state_dict":model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(check_point)
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        # print some examples to a folder
        save_prediction_as_imgs(
            val_loader, model, folder = "saved_images/", device = DEVICE
        )



if __name__ == '__main__':
    main()




