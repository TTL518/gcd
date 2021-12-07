from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageReader(Dataset):

    def __init__(self,
                 data_path,
                 data_name,
                 data_type,
                 crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((252, 252)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)
    