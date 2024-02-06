import cv2
import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from utils.model import ResNet34
from torchvision import transforms

classify = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])

path = 'dataset/pred'
image_path_list = os.listdir(path)
image_path_list_10 = random.sample(image_path_list, 10)

image_list = []
image_tensor_list = []
for i in image_path_list_10:
    img = cv2.imread(os.path.join(path, i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_list.append(img)
    img = Image.fromarray(img)
    img = transform(img)
    image_tensor_list.append(img)
image = torch.stack(image_tensor_list, 0)

net = ResNet34(6)
net.load_state_dict(torch.load('model_weights/ResNet34.pth'))
pred = torch.argmax(net(image), dim=1)

pred_list = []
for i in pred:
    pred_list.append(classify[int(i)])
print(pred_list)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.imshow(image_list[i])
plt.show()
