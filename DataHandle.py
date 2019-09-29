from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5],std=[0.5])
])

class DATA_SET(Dataset):
    def __init__(self, PATH):
        self.PATH = PATH
        self.img_list =[]
        self.img_list.extend(os.listdir(PATH))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.PATH,self.img_list[index]))
        img_data = transforms(img)
        return img_data

if __name__ == '__main__':
    data = DATA_SET(r"G:\项目\20190830\faces")

    print(data[1])