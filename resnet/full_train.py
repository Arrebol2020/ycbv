import copy
import random
import time
import numpy as np
import timm
import torch
import torchvision
import os
from torchvision import transforms
import torch.utils.data
import torch.nn

from resnet.utils.egine import train_one_epoch, valid_one_epoch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "epoch": 30,
    "num_classes": 21,
    "batch_size": 32,
    "num_workers": 4,
    "optimizer": "Adam",
    "lr": 0.0003,
    "model_save": os.path.join(BASE_DIR, "data", "model"),
    "verbose_step": 1
}


def set_seed(seed=2021):
    """
    固定随机种子，便于复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def get_model(pretrained=False):
    if pretrained:
        net = torchvision.models.resnet50(pretrained=pretrained)
    else:
        net = torchvision.models.resnet50(pretrained=pretrained)
        # 加载权重
        model_weight_path = os.path.join(BASE_DIR, "data", "pretrain_model", "resnet50_caffe.pth")
        net.load_state_dict(torch.load(model_weight_path))

    # 修改模型最后一层
    in_features = net.fc.in_features
    net.fc = torch.nn.Linear(in_features, CFG["num_classes"])

    return net


def freeze_conv(net):
    """
    冻结rennet除了全连接层以外的参数
    :param net:
    :return:
    """
    for param in net.parameters():  # 对所有参数冻结
        param.requires_grad = False

    for param in net.fc.parameters():  # 对全连接层的参数取消冻结
        param.requires_grad = True


if __name__ == "__main__":
    set_seed()  # 设置种子

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """
    ImageFolder():
    root: 数据的根路径
    transform: 图像变化，以PIL 图像为输入的变化
    target_transform: 接受目标并对其进行转换的函数/转换。
    """
    train_data_path = os.path.join(BASE_DIR, "data", "train")
    test_data_path = os.path.join(BASE_DIR, "data", "test")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_data = torchvision.datasets.ImageFolder(root=train_data_path,
                                                  transform=data_transform["train"])
    test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                                 transform=data_transform["test"])

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=CFG["batch_size"],
                                               shuffle=True,
                                               num_workers=CFG["num_workers"])
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=CFG["batch_size"],
                                              shuffle=False,
                                              num_workers=CFG["num_workers"])
    """
    # 查看dataloader的数据
    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()
    print(type(images), type(labels))
    print(len(images), len(labels))
    print(images[0].shape)
    print(labels)
    """
    # 获取模型
    model = get_model(pretrained=True)
    """
    model = timm.create_model('resnet50', pretrained=True)
    # 修改模型最后一层
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, CFG["num_classes"])
    model.to(device)
    """
    model.to(device)
    print('train device: ', device)

    # 训练
    loss_fc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if CFG["optimizer"] == "Adam":
        pass
    else:
        raise Exception("请添加{}的优化器", CFG["optimizer"])

    best_acc = 0.0
    # best_weights = copy.deepcopy(model.state_dict())

    begin_time = time.time()

    for epoch in range(CFG["epoch"]):

        train_loss, train_acc = train_one_epoch(epoch=epoch,
                                                model=model,
                                                verbose_step=CFG["verbose_step"],
                                                data_loader=train_loader,
                                                device=device,
                                                criterion=loss_fc,
                                                opt=optimizer,
                                                )

        test_loss, test_acc = valid_one_epoch(epoch=epoch,
                                              model=model,
                                              data_loader=test_loader,
                                              device=device,
                                              criterion=loss_fc
                                              )
        scheduler.step()

        if test_acc > best_acc:  # 更新权重
            best_acc = test_acc
            # best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(CFG["model_save"], "resnet50.pth"))

    time_elapsed = time.time() - begin_time  # 模型训练所花费的时间
    print(f'Training Complete in {time_elapsed // 60}m, {time_elapsed % 60}s')
    print(f'Best val acc: {best_acc: .4f}')


