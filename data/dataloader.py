import torch
from torchvision import datasets
from torchvision import transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageNet("../datasets/ILSVRC/Data/DET/test", 
                      train="true",
                      download=True,
                      transforms= transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                      ])
                    )
)