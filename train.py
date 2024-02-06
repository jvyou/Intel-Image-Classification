import torch
import torch.nn as nn
from utils.dataLoader import load_data
from utils.model import ResNet34
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def accuracy(predictions, labels):
    pred = torch.argmax(predictions, 1)
    rights = (pred == labels).sum().float()
    return rights, len(labels)


def train(net, epochs, train_iter, test_iter, device, loss, optimizer, model_path, auto_save):
    train_acc_list = []
    test_acc_list = []

    train_loss_list = []
    test_loss_list = []

    net = net.to(device)

    for epoch in range(epochs):
        net.train()
        train_rights = 0
        train_loss = 0
        train_len = 0
        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc="train epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_rights += accuracy(y_hat, y)[0]
                train_len += accuracy(y_hat, y)[1]
                train_loss += l.detach()
                pbar.set_postfix({'loss': "{:.6f}".format(train_loss / train_len),'acc':"{:.6f}".format(train_rights / train_len)})
                pbar.update(1)
            train_acc_list.append(train_rights.cpu().numpy() / train_len)
            train_loss_list.append(train_loss.cpu().numpy() / train_len)

        net.eval()
        test_rights = 0
        test_loss = 0
        test_len = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc="test epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_rights += accuracy(y_hat, y)[0]
                test_len += accuracy(y_hat, y)[1]
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    pbar.set_postfix({'loss': "{:.6f}".format(test_loss / test_len),'acc':"{:.6f}".format(test_rights / test_len)})
                    pbar.update(1)
            test_acc_list.append(test_rights.cpu().numpy() / test_len)
            test_loss_list.append(test_loss.cpu().numpy() / test_len)

        if (epoch + 1) % auto_save == 0:
            torch.save(net.state_dict(), model_path)

    plt.subplot(211)
    plt.plot([i+1 for i in range(len(train_acc_list))], train_acc_list, 'bo--', label="train_acc")
    plt.plot([i+1 for i in range(len(test_acc_list))], test_acc_list, 'ro--', label="test_acc")
    plt.title("train_acc vs test_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(212)
    plt.plot([i+1 for i in range(len(train_loss_list))], train_loss_list, 'bo--', label="train_loss")
    plt.plot([i+1 for i in range(len(test_loss_list))], test_loss_list, 'ro--', label="test_loss")
    plt.title("train_loss vs test_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.savefig('logs/acc and loss.png')
    plt.show()


if __name__ == '__main__':
    batch_size = 128  # 批量大小
    crop_size = 64  # 裁剪大小
    in_channels = 3  # 输入图像通道
    classes_num = 6  # 输出标签类别
    num_epochs = 100  # 总轮次
    auto_save = 10  # 自动保存的间隔轮次
    lr = 1e-3  # 学习率
    weight_decay = 1e-4  # 权重衰退
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择设备

    classify = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
    train_iter, test_iter = load_data(batch_size, crop_size, classify)

    net = ResNet34(classes_num)  # 定义模型
    model_path = 'model_weights/ResNet34.pth'

    loss = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # 定义优化器

    print("训练开始")
    time_start = time.time()
    train(net, num_epochs, train_iter, test_iter, device=device, loss=loss, optimizer=optimizer, model_path=model_path, auto_save=auto_save)
    torch.save(net.state_dict(), model_path)
    time_end = time.time()
    seconds = time_end - time_start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("训练结束")
    print("本次训练时长为：%02d:%02d:%02d" % (h, m, s))
