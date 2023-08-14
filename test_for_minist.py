import torch
import torchvision
from torchvision import transforms
import numpy as np
from DShap import DShap

# 加载预训练的 ResNet-18 模型，不包含最后一层
resnet_model = torchvision.models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # 去掉最后一层

# 添加一个全局平均池化层
resnet_model.add_module('GlobalAvgPool', torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

# MNIST 数据预处理（包括通道转换）
transform = transforms.Compose([
    transforms.Resize(224),  # 调整大小
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道的灰度图像
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 加载 MNIST 数据集
mnist_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=1, shuffle=False)

# 特征提取
features = []
labels = []
i = 0
for images, label in data_loader:
    with torch.no_grad():
        output = resnet_model(images)
        features.append(output.squeeze().numpy())
        labels.append(np.array(label))
        # print(features[0].shape)
        # print(label)
    i = i + 1
    # if i==500:
    #   break

# 将特征合并为一个数组
features_array = np.array(features)
labels = np.array(labels)
np.save('feature.npy', features_array)
np.save('labels.npy', labels)

# 打印特征数组的形状
print(features_array.shape)  # 应该是 (60000, 128) 或类似的形状

train_size = int(i * 0.8)
X_raw = np.load("feature.npy")
y_raw = np.load("labels.npy")
print(y_raw.tolist())
y_raw = y_raw.tolist()

y_temp = []
for i in range(len(y_raw)):
    y_temp.append(y_raw[i][0])
y_raw = np.array(y_temp)
# 加载MNIST数据集
X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]

model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test,
              sources=None,
              sample_weight=None,
              model_family=model,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)
dshap.run(100, 0.1, g_run=False)

model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=1)
dshap.run(100, 0.1)

dshap.merge_results()

dshap.performance_plots([dshap.vals_tmc, dshap.vals_g, dshap.vals_loo], num_plot_markers=20,
                        sources=dshap.sources)
