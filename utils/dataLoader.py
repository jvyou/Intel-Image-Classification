import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image


def get_mean_std(dl):
    sum_, squared_sum, batches = 0, 0, 0
    for data, _ in dl:
        sum_ += torch.mean(data, dim=([0, 2, 3]))
        squared_sum += torch.mean(data ** 2, dim=([0, 2, 3]))
        batches += 1

    mean = sum_ / batches
    std = (squared_sum / batches - mean ** 2) ** 0.5
    return mean, std


class InterDataset(Dataset):
    def __init__(self, root, transform, classify):
        super(InterDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.classify = classify

        self.filenames = []
        self.labels = []

        labels = os.listdir(self.root)
        for i in labels:
            filenames = os.listdir(os.path.join(self.root, i))
            for j in filenames:
                self.filenames.append(os.path.join(self.root, i, j))
                self.labels.append(i)

    def __getitem__(self, index):
        image_name = self.filenames[index]
        label = self.labels[index]

        label_index = self.classify[label]
        label_index = torch.tensor(label_index)

        image = Image.open(image_name)
        image = self.transform(image)

        return image,label_index

    def __len__(self):
        return len(self.filenames)


def load_data(batch_size, crop_size, classify):
    stats = ((0.0017, 0.0018, 0.0018), (0.0010, 0.0010, 0.0011))
    train_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats,inplace=True)])
    test_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats,inplace=True)])

    train_iter = torch.utils.data.DataLoader(InterDataset('./dataset/train',train_transform,classify), batch_size, shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(InterDataset('./dataset/test',test_transform,classify), batch_size, shuffle=False, drop_last=True)

    return train_iter, test_iter


if __name__ == '__main__':
    classify = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
    train_iter, test_iter = load_data(64, 64, classify)
    for i, (X, y) in enumerate(train_iter):
        print(X[0].shape,y[0])
        break
    mean, std = get_mean_std(train_iter)
    print(mean,std)
