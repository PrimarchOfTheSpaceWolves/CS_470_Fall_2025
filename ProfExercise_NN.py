import cv2
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import decode_image, read_image
from torchvision.transforms import v2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

class DogCatDataset(Dataset):
    def __init__(self, root, train, transform, seed=42):
        self.root = root
        self.transform = transform
        
        cat_dir = os.path.join(root, "Cat")
        dog_dir = os.path.join(root, "Dog")
        
        cat_files = os.listdir(cat_dir)
        dog_files = os.listdir(dog_dir)
        
        all_files = []
        for filename in cat_files:
            if ".jpg" in filename:
                all_files.append(os.path.join("Cat", filename))
        for filename in dog_files:
            if ".jpg" in filename:
                all_files.append(os.path.join("Dog", filename))
        
        all_files = np.array(all_files)                
        train_files, test_files = train_test_split(all_files, 
                                                   test_size=0.3, 
                                                   random_state=seed)
        
        if train:
            self.chosen_files = train_files
        else:
            self.chosen_files = test_files
            
    def __len__(self):
        return len(self.chosen_files)
    
    def __getitem__(self, index):
        filepath = self.chosen_files[index]
        print(filepath)
        
        if "Cat" in filepath:
            label = 0
        else:
            label = 1
            
        fullpath = os.path.join(self.root, filepath)
        image = decode_image(fullpath)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)        
    
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(3072, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )              
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits
    

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    
    device = "cuda"
    
    model = SimpleNetwork().to(device)
    print(model)
    count_parameters(model)
    
    base_data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Resize(size=(320,320))
    ])
    
    data_aug_transform = v2.Compose([
        base_data_transform,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(
            degrees=45, 
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    ])
    
    #train_ds = DogCatDataset(root="./data/PetImages", 
    #                             train=True, 
    #                             transform=data_transform)
    
    train_ds = datasets.CIFAR10(root="./data", 
                                train=True,
                                transform=data_aug_transform,
                                download=True)
    
    train_noaug_ds = datasets.CIFAR10(root="./data", 
                                train=True,
                                transform=base_data_transform,
                                download=True)
    
    test_ds = datasets.CIFAR10(root="./data", 
                                train=False,
                                transform=base_data_transform,
                                download=True)    

    batch_size = 64
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True)
    train_noaug_dataloader = DataLoader(train_noaug_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)
    
    #for (index, batch) in enumerate(train_dataloader):
    data_iter = iter(train_dataloader)
    for _ in range(5):
        batch = next(data_iter)
    
        image = batch[0]
        label = batch[1]
        
        image = image[0]
        label = label[0]
        
        image = image.numpy()
        label = label.numpy()
        
        image = np.transpose(image, [1,2,0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=None, fx=5.0, fy=5.0)
        
        cv2.imshow("IMAGE", image)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()        
            
    epoch_cnt = 5
    
    #for epoch in range(epoch_cnt):    
    #    for batch in train_dataloader:
        
    
if __name__ == "__main__":
    main()
    