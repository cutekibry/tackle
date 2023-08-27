# tackle
tackle 是本人 AI 派第二轮测试的项目仓库，选题为“CV 构架分析”。

## 概述
同目录下共有五个文件夹，四个文件夹分别对应四篇 paper，剩下一个文件夹 `data` 存放数据。每个子文件夹下的 `learning` 文件夹是学习笔记，`implementation` 文件夹是复现代码。

学习笔记理论基本阅读完毕，实验阅读了部分。

复现受机能和网络限制，受到很大阻碍，仅实现了：

* ResNet 论文中 ImageNet 的图像分类测试（使用 BIRDS-525 替代 ImageNet 进行测试）。
  * 由于 ImageNet 较大（144.3G），只能使用 [BIRDS 525 SPECIES- IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) 数据集作为 ImageNet 的替代，验证模型效果。该数据集包含 $8\ 4635$ 张训练图、$2625$ 张测试图、$2625$ 张验证图，每张图为 $224 \times 224$ 大小的 RGB 图像，代表了 $525$ 种鸟类的其中一种。该数据集使用 CC0 协议。
  * 受到机能限制，BIRDS-525 的复现参数做了大量调整，导致实验结果和论文不符，难以对此进行进一步的探究。
* ResNet 论文中 CIFAR-10 的图像分类测试。
  * 按原论文参数复现了除 ResNet-1202（计算量过大不进行复现）外的其他模型，结果基本符合原文结论。可以观测到层数在 $20 \sim 110$ 之间时，层数越多，朴素 CNN 表现越差，ResNet 表现越好（即 ResNet 对退化现象的抑制作用），且 ResNet 表现优于朴素 CNN。
* ConvNeXt 论文中的图像处理方式和模型。
  * 受机能限制不进行训练和测试。

对于总结和提出新构架：暂时无法提出创新的构架。CV 是实践的学科，即使提出了创新的构想，实践也可能表现不优秀；而目前模型训练均要求较高的算力（即使是 FLOPs 较小的 MobileNet V3 也对训练配置有极高的要求），不实践可能只是纸上谈兵。因而无法提出构架。

机器配置：

* 系统：`Kubuntu 22.04.2 LTS x86_64`
* CPU：`12th Gen Intel i7-12700H (20) @ 4.600GHz`
* GPU：`NVIDIA GeForce RTX 3060 Mobile / Max-Q`
* 内存：16GB

## 环境要求
需要安装的包：

* `pytorch`
* `pytorch-model-summary`
* `seaborn`
* `scipy`

代码默认使用单个 `CUDA` 支持的 GPU 进行运算，可能需要一定更改才能在其他机器上运行。