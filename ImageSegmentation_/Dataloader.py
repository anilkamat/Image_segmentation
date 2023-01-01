import os
import numpy as np
from PIL import Image
import torch.utils.data as Dataset


class CarvanaDataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform= None):
        super(CarvanaDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, indx):
        image_path = os.path.join(self.image_dir,self.images[indx])
        mask_path = os.path.join(self.mask_dir,self.images[indx].replace('.jpeg','_mask.gif'))
        image = np.arrary(Image.open(image_path).convert('RGB'))
        mask = np.arrary(Image.open(mask_path).convert('L'), dtype = np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None: 
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


