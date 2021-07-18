import time
from tqdm import tqdm
import torch


def train_one_epoch(epoch, model, verbose_step, data_loader, device, criterion, opt):
    model.train()

    train_loss = 0.
    train_acc = 0.

    begin_time = time.time()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        opt.zero_grad()

        predicts = model(images)
        loss = criterion(predicts, labels)
        loss.backward()
        opt.step()

        preds = torch.argmax(predicts, 1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        train_loss += loss.item()*images.shape[0]
        train_acc += sum(preds == targets)

        if ((step + 1) % verbose_step == 0) or ((step + 1) == len(data_loader)):
            description = f'epoch {epoch} loss: {loss.item()*images.shape[0]:.4f}'

            pbar.set_description(description)
    epoch_loss = train_loss / len(data_loader)
    epoch_acc = train_acc/len(data_loader)

    time_elapsed = time.time() - begin_time  # 训练一次所花费的时间

    print(f'train epoch {epoch} loss: {epoch_loss:.4f}, acc: {epoch_acc: .4f} '
          f'time: {time_elapsed // 60}m {time_elapsed % 60}s')

    return epoch_loss, epoch_acc


def valid_one_epoch(epoch, model, data_loader, device, criterion):
    model.eval()

    val_loss = 0.
    val_acc = 0.

    begin_time = time.time()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predicts = model(images)
            loss = criterion(predicts, labels)

            preds = torch.argmax(predicts, 1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

        val_loss += loss.item() * images.shape[0]
        val_acc += sum(preds == targets)

    epoch_loss = val_loss / len(data_loader)
    epoch_acc = val_acc / len(data_loader)

    time_elapsed = time.time() - begin_time  # 测试一次所花费的时间

    print(f'valid epoch {epoch} loss: {epoch_loss:.4f}, acc: {epoch_acc: .4f} '
          f'time: {time_elapsed // 60}m {time_elapsed % 60}s')

    return epoch_loss, epoch_acc
