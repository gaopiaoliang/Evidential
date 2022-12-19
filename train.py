import os
import torch
import random
import argparse
import numpy as np
import scipy.io as sio
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from evidential_fusion import EFN
from dataset import Multi_view_data
import warnings
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    # epoch=500
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--view', type=int, default=2,
                        help='number of sense view')
    parser.add_argument('--classes', type=int, default=11,
                        help='number of class')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # optimization
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--lr_decay_epochs', type=str, default='40,80,120,160,200',
                        help='where to decay lr, can be a list，if is None，no lr_dacey!')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay for weight_decay')

    # model dataset

    opt = parser.parse_args(args=[])

    if opt.lr_decay_epochs is not None:
        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

    if opt.seed is not None:
        seed_it(opt.seed)
        warnings.warn(
            'You have chosen to seed training.This will turn on the CUDNN deterministic setting,which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')

    return opt
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed) #numpy
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, epoch, optimizer, train_loader):
    model.train()
    loss_meter = AverageMeter()
    sum_loss = 0.0
    data_num, correct_num = 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        data_num += target.size(0)
        target = Variable(target.long().cuda())
        optimizer.zero_grad()
        evidences, evidence_a, loss = model(data, target)
        # compute gradients and take step
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        sum_loss += loss.item()
        train_loss = sum_loss / (batch_idx + 1)
        _, predicted = torch.max(evidence_a.data, 1)
        correct_num += (predicted == target).sum().item()
        train_acc = correct_num / data_num
    print(f"epoch{epoch + 1}")
    print(f"train: loss:{'%.2f'%train_loss} acc:{'%.4f'%train_acc}")
    return train_loss, train_acc


def evalute(model, loader):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            target = Variable(target.long().cuda())
            evidences, evidence_a, loss = model(data, target)
            loss_meter.update(loss.item())
            sum_loss += loss.item()

            val_loss = sum_loss / (batch_idx + 1)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            val_acc = correct_num / data_num


    return val_loss, val_acc
def main():
    opt = parse_option()
    if torch.cuda.is_available():
        print('gpu个数:', torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print('gpu名称:', torch.cuda.get_device_name(idx))
    print(opt)
    # data loading
    data_name = 'vgg-airound'
    data_path = 'datasets/' + data_name
    dims = [[4096], [4096]]
    view_num = opt.view
    classes = opt.classes
    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(data_path,'train',view_num), batch_size = opt.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        Multi_view_data(data_path, 'val', view_num), batch_size=opt.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(data_path, 'test', view_num), batch_size=opt.batch_size, shuffle=False)

    # model loading
    model = EFN(classes, dims, view_num)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # training
    best_acc = 0.0
    for epoch in range(opt.epochs):
        train_loss, train_acc = train(model, epoch, optimizer, train_loader)
        val_loss, val_acc = evalute(model, val_loader)
        print(f"val: loss:{'%.2f' % val_loss}  acc:{'%.4f' % val_acc}")
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
        torch.save(model, 'EFN_best.pth')
    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model1 = (torch.load('EFN_best.pth'))
    print('loaded from ckpt!')

    test_loss, test_acc = evalute(model1, test_loader)
    print(f"test: loss:{'%.2f' % test_loss} acc:{'%.4f' % test_acc}")
if __name__=="__main__":
    main()


