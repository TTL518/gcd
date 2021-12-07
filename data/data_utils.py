import os

import torch
from PIL import Image


def read_txt(path, data_num):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        if data_num == 2:
            data_1, data_2 = line.split()
        else:
            data_1, data_2, data_3, data_4, data_5 = line.split()
            data_2 = [data_2, data_3, data_4, data_5]
        data[data_1] = data_2
    return data


def process_isc_data(data_path):
    if not os.path.exists('{}/uncropped'.format(data_path)):
        os.mkdir('{}/uncropped'.format(data_path))
    train_images, query_images, gallery_images = {}, {}, {}
    for index, line in enumerate(open('{}/Eval/list_eval_partition.txt'.format(data_path), 'r', encoding='utf-8')):
        if index > 1:
            img_name, label, status = line.split()
            img = Image.open('{}/Img/{}'.format(data_path, img_name)).convert('RGB')
            save_name = '{}/uncropped/{}_{}'.format(data_path, img_name.split('/')[-2], os.path.basename(img_name))
            img.save(save_name)
            if status == 'train':
                if label in train_images:
                    train_images[label].append(save_name)
                else:
                    train_images[label] = [save_name]
            elif status == 'query':
                if label in query_images:
                    query_images[label].append(save_name)
                else:
                    query_images[label] = [save_name]
            else:
                if label in gallery_images:
                    gallery_images[label].append(save_name)
                else:
                    gallery_images[label] = [save_name]

    torch.save({'train': train_images, 'query': query_images, 'gallery': gallery_images},
               '{}/uncropped_data_dicts.pth'.format(data_path))
