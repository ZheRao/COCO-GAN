import os
from PIL import Image
from torch.utils.data import Dataset
class image_dataset(Dataset):
    def __init__(self,image_folder,transforms=None):
        super().__init__()
        self.image_names = list(os.listdir(image_folder))
        self.image_path = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path,self.image_names[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image
