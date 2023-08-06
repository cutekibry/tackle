本人选题为“CV 构架分析”。

## 概述
同目录下共有五个文件夹，四个文件夹分别对应四篇 paper，剩下一个文件夹 `data` 存放数据。每个子文件夹下的 `learning` 文件夹是学习笔记，`implementation` 文件夹是复现代码。

### ResNet
阅读笔记：基本阅读完毕。

复现（ImageNet）：由于 ImageNet 较大（50~110G），且网络限制 / 机能不足，只能使用 [BIRDS 525 SPECIES- IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) 数据集作为 ImageNet 的替代，验证模型效果。该数据集包含 $8\ 4635$ 张训练图、$2625$ 张测试图、$2625$ 张验证图，每张图为 $224 \times 224$ 大小的 RGB 图像，代表了 $525$ 种鸟类的其中一种。该数据集使用 CC0 协议。

受到机能限制，BIRDS-525 的复现参数做了不少调整，导致结果不令人满意，只能观测到 ResNet 比同等复杂度的朴素 CNN 更优，但不能观测到 ResNet 对退化现象的抑制作用。

复现（CIFAR-10）：基本按原论文参数进行复现，得到的结果和准确率也基本符合原文结论，可以观测到层数在 $20 \sim 56$ 之间时，层数越多，朴素 CNN 表现越差，ResNet 表现越好（即 ResNet 对退化现象的抑制作用），且 ResNet 表现优于朴素 CNN。

### ViT
阅读笔记：基本阅读完毕。

复现：由于最小上游数据集就是 1T 大小的 ImageNet-21K，加上 ViT 本身需要大量的上游数据才能在下游任务有优秀的表现，机能和网络基本不允许复现，就不复现了。

### ConvNeXt
阅读笔记：

## 环境要求
需要安装的包：

* `pytorch`
* `pytorch-model-summary`
* `seaborn`
* `scipy`

代码默认使用 `CUDA` 支持的 GPU 进行运算，可能需要一定更改才能在其他机器上运行。