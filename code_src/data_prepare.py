from torch.utils.data import Dataset
import cv2 as cv


# def image dataset
class myDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv.resize(self.images[idx], (256, 256))
        image = self.transform(image)
        
        return image


