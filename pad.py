import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
import numpy
import csv
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
from resnet import resnet34
import random
BATCH_SIZE = 32


# 定义网络模型亦即Net 这里定义一个简单的全连接层784->10

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(2560, 640)
        self.fc2 = nn.Linear(640, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        # x = self.bn0(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 2560)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.fc4(x)
        return x


def train_collate_1(batch):
    tf = transforms.Compose(
        [transforms.Pad(2, fill=0),
        transforms.ToTensor()
         ])
    return train_collate(batch, tf)


def train_collate_2(batch):
    tf = transforms.Compose(
        [transforms.Pad(2, fill=0),
        transforms.RandomHorizontalFlip(0.2),
         transforms.RandomVerticalFlip(0.2),
         transforms.RandomApply([transforms.RandomRotation(45)], 0.2),
         transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5)], 0.2),
         transforms.ToTensor()
         ])
    return train_collate(batch, tf)


def train_collate_3(batch):
    tf = transforms.Compose(
        [transforms.Pad(2, fill=0),
        transforms.RandomHorizontalFlip(0.3),
         transforms.RandomVerticalFlip(0.3),
         transforms.RandomApply([transforms.RandomRotation(60)], 0.4),
         transforms.RandomApply([transforms.ColorJitter(0.6, 0.6, 0.6)], 0.4),
         transforms.ToTensor()
         ])
    return train_collate(batch, tf)


def train_collate(batch, tf):
    transform_data = []
    labels = []
    for (imgnp, label) in batch:
        imgnp = imgnp.reshape(28, 28)
        img = Image.fromarray(imgnp, 'L')
        img = tf(img)
        transform_data.append(numpy.array(img).reshape(-1))
        labels.append(label)
    transform_data = numpy.array(transform_data)
    transform_data = torch.from_numpy(transform_data)
    labels = torch.from_numpy(numpy.array(labels))
    return [transform_data, labels]


def test_collate(batch):
    tf = transforms.Compose(
        [transforms.Pad(2, fill=0),transforms.ToTensor()])
    transform_data = []
    labels = []
    for (imgnp, label) in batch:
        imgnp = imgnp.reshape(28, 28)
        img = Image.fromarray(imgnp, 'L')
        img = tf(img)
        transform_data.append(numpy.array(img).reshape(-1))
        labels.append(label)
    transform_data = numpy.array(transform_data)
    transform_data = torch.from_numpy(transform_data)
    labels = torch.from_numpy(numpy.array(labels))
    return [transform_data, labels]


def final_test_collate(batch):
    tf = transforms.Compose(
        [transforms.Pad(2, fill=0),transforms.ToTensor()])
    transform_data = []
    for imgnp in batch:
        imgnp = imgnp.reshape(28, 28)
        img = Image.fromarray(imgnp, 'L')
        img = tf(img)
        transform_data.append(numpy.array(img).reshape(-1))
    transform_data = numpy.array(transform_data)
    transform_data = torch.from_numpy(transform_data)
    return transform_data


def train_model(modelclass, train_data, train_label, threshold):
    model = modelclass().cuda() #实例化卷积层
    loss = nn.CrossEntropyLoss().cuda()  # 损失函数选择，交叉熵函数
    lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.1)
    train_dataset = [(X_train[index], y_train[index]) for index in range(len(X_train))]
    test_dataset = [(X_test[index], y_test[index]) for index in range(len(X_test))]

    train_stage = 1
    last_stage_epoch = 0

    # 加载小批次数据，即将数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_collate_1)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_collate)

    num_epochs = 50

    modelnum = get_model_num()
    best_test_acc = 0
    best_epoch_num = 0

    epoch = 0

    while epoch < num_epochs:
        train_loss = 0  # 定义训练损失
        train_acc = 0  # 定义训练准确度
        model.train()  # 将网络转化为训练模式
        for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
            X = X.view(-1, 1, 32, 32)  # X:[64,1,28,28] -> [64,784]将X向量展平
            X = Variable(X).float().cuda()          #包装tensor用于自动求梯度
            label = Variable(label).long().cuda()
            out = model(X)  # 正向传播
            lossvalue = loss(out, label)  # 求损失值
            optimizer.zero_grad()  # 优化器梯度归零
            lossvalue.backward()  # 反向转播，刷新梯度值
            optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数
            # 计算损失
            train_loss += float(lossvalue)
            # 计算精确度
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            train_acc += acc

        newloss = train_loss / len(train_loader)
        newacc = train_acc / len(train_loader)

        print("echo:" + ' ' + str(epoch))
        print("lose:" + ' ' + str(newloss))
        print("accuracy:" + ' ' + str(newacc))

        model.eval()
        test_acc = 0
        for i, (X, label) in enumerate(test_loader):
            X = X.view(-1, 1, 32, 32)
            X = Variable(X).float().cuda()
            label = Variable(label).long().cuda()
            testout = model(X)
            _, pred = testout.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            test_acc += acc
        test_acc = test_acc / len(test_loader)
        print("test acc: " + str(test_acc))
        save_model(model, modelnum, epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch_num = epoch

        if (newacc > test_acc and test_acc > 0.8 + train_stage*0.04) or epoch - last_stage_epoch > 15:
            train_stage += 1
            print("train stage changed: "+str(train_stage))
            epoch = best_epoch_num
            model = load_model(modelclass, modelnum, best_epoch_num)
            last_stage_epoch = epoch
            best_test_acc = 0
            
            if train_stage == 2:
                new_train_collate = train_collate_2
                lr = lr / 5
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif train_stage == 3:
                lr = lr / 2
                optimizer = optim.SGD(model.parameters(), lr=lr)
                new_train_collate = train_collate_3
            else:
                break
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=new_train_collate)

        if test_acc > threshold:
            break
        epoch += 1

    save_model(model, modelnum)

    return model


def load_model(modelclass, modelnum, beginepoch):
    model = modelclass().cuda()
    path_prefix = "./data/model"
    path_suffix = ".pt"
    path = path_prefix + str(modelnum) + "epoch" + str(beginepoch) + path_suffix
    model.load_state_dict(torch.load(path))
    return model


def get_mean_std(dataset):
    """
    计算数据集的均值和方差
    """
    tf = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize([],[])
         ])

    means = 0
    stdevs = 0
    for data in dataset:
        data = data.reshape(28, 28)
        img = Image.fromarray(data, 'L')
        img = tf(img)
        data = numpy.array(img).reshape(-1)
        means += data.mean()
        stdevs += data.std()

    means = numpy.asarray(means) / len(dataset)
    stdevs = numpy.asarray(stdevs) / len(dataset)

    print("normMean = {}".format(means))
    print("normstdevs = {}".format(stdevs))


def get_model_num():
    path_prefix = "./data/model"
    path_suffix = "epoch0.pt"
    path_suffix1 = ".pt"
    modelnum = 0
    path = path_prefix + str(modelnum) + path_suffix
    path1 = path_prefix + str(modelnum) + path_suffix1
    while os.path.exists(path) or os.path.exists(path1):
        modelnum += 1
        path = path_prefix + str(modelnum) + path_suffix
        path1 = path_prefix + str(modelnum) + path_suffix1
    return modelnum


def save_model(model, modelnum, epoch=None):
    path_prefix = "./data/model"
    path_suffix = ".pt"
    if epoch is not None:
        path = path_prefix + str(modelnum) + "epoch" + str(epoch) + path_suffix
    else:
        path = path_prefix + str(modelnum) + path_suffix
    torch.save(model.state_dict(), path)
    print("model saved! path: " + path)



if __name__ == "__main__":
    train_data = numpy.load("./data/train.npy")
    origin_train_label = []
    with open("./data/train.csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            origin_train_label.append(int(row[1]))

    train_data = numpy.array(train_data)
    train_label = numpy.array(origin_train_label)

    from googlenet import googlenet
    model = train_model(resnet34, train_data, train_label, 0.95)
    # model = continue_train_model(resnet34, 1,9,train_data,train_label,0,4)

    test_data = numpy.load("./data/test.npy")
    final_test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=final_test_collate)
    index = 0
    with open("./data/submit.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_id', 'label'])
        for X in final_test_loader:
            X = X.view(-1, 1, 32, 32)
            X = Variable(X).float().cuda()
            testout = model(X)
            _, pred = testout.max(1)
            for p in pred.cpu().numpy().tolist():
                csv_writer.writerow([index, p])
                index = index + 1
