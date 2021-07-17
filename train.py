import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from PIL import Image
from xml.dom.minidom import parse


class YCBVDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.filenames = self.get_dataset_info()
        self.xml_dir = os.path.join(root, "Annotations")
        self.img_dir = os.path.join(root, "Images")

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.img_dir, self.filenames[idx].replace('\n', '') + ".png")
        bbox_xml_path = os.path.join(self.xml_dir, self.filenames[idx].replace('\n', '') + ".xml")
        img = Image.open(img_path).convert("RGB")

        dom = parse(bbox_xml_path)  # 读取xml文件
        data = dom.documentElement  # 获取文档元素对象
        objects = data.getElementsByTagName('object')  # 获取 objects
        boxes = []  # object的bounding box
        labels = []  # 对应的标签
        areas = []  # 面积列表
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 获取label
            labels.append(int(name[-1]))  # 背景的label是0

            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        areas = [(boxes[i][3] - boxes[i][1]) * (boxes[i][2] - boxes[i][0])
                 for i in range(len(boxes))]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        areas = torch.as_tensor(areas, dtype=torch.float32)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的
            # transforms.py里就有RandomHorizontalFlip时target变换的示例
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.filenames) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.root))
        return len(self.filenames)

    def get_dataset_info(self):
        if "train" == self.mode:
            txt_path = os.path.join(root, "ImageSets", "Main", "train.txt")
        elif "test" == self.mode:
            txt_path = os.path.join(root, "ImageSets", "Main", "val.txt")
        else:
            raise Exception("self.mode 无法识别，仅支持(train, test)")

        filenames = []
        with open(txt_path, 'r') as f:
            filenames = f.readlines()

        return filenames


def get_object_detection_model(num_classes):
    """
    生成模型实例
    :param num_classes: 分类数+1
    :return:
    """
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


import tools.utils
import tools.transforms as T
from tools.engine import train_one_epoch, evaluate
# utils、transforms、engine就是刚才下载下来的utils.py、transforms.py、engine.py


def get_transform(mode=None):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if "train" == mode:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


if __name__ == "__main__":

    import tools.utils

    root = './data/VOC2007'

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 背景+21类
    num_classes = 22
    # 初始化dataset
    train_dataset = YCBVDataset(root, mode="train", transforms=get_transform(mode="train"))
    test_dataset = YCBVDataset(root, mode="test", transforms=get_transform(mode="test"))

    # 定义训练和测试的data loader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4,
        collate_fn=tools.utils.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=tools.utils.collate_fn)

    # 初始化模型
    model = get_object_detection_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # Adam 优化器
    optimizer = torch.optim.Adam(params, lr=0.0003, weight_decay=0.0005)

    # cos学习策略
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        tools.engine.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        tools.engine.evaluate(model, test_data_loader, device=device)

        print('')
        print('==================================================')
        print('')

    print("That's it!")
    torch.save(model, './ycbv.pkl')

