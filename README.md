# Generative Adversarial Network (GAN) | 生成对抗网络

### 选择语言 | Language

[中文简介](#简介) | [English](#introduction)

### 结果 | Result

**实战 1：MNIST。** 判别器损失从 1.997 降到 1.381，生成器损失从 0.798 降到 0.702。D(real) 从 0.561 降到 0.503，D(fake) 从 0.465 升到 0.497，两边在 0.5 会合。纳什均衡时判别器损失约为 log(4) ≈ 1.386，生成器的 non-saturating 损失约为 0.693。这一轮已经停在均衡附近。第 1 个 epoch 还是噪点，第 10 个 epoch 能看出数字，第 20 个 epoch 的 0–9 轮廓清楚。

固定噪声的生成过程（`runs/mnist_gan/samples_progress.png`）：

<img width="1764" height="588" alt="image" src="https://github.com/user-attachments/assets/3a2a2104-8610-4709-a80b-50acee211a27" />


训练曲线（`runs/mnist_gan/loss.png`）：

<img width="1330" height="476" alt="image" src="https://github.com/user-attachments/assets/1782f24d-8f5f-4ff0-b7e1-78a9f416daec" />


测试集上的判别器打分，以及真实测试图与生成图对照（`runs/mnist_gan/test_eval.png`）：

<img width="1288" height="504" alt="image" src="https://github.com/user-attachments/assets/4799302d-5295-4ae6-88d7-e8f7c9a60743" />


**实战 2：CIFAR-10。** 生成器损失从 1.746 降到 0.854，判别器损失从约 1.05 升到 1.305。D(real) 从 0.664 降到 0.545，D(fake) 从 0.330 升到 0.455，还在向 0.5 靠近，20 个 epoch 没有完全到均衡。第 1 个 epoch 是色块噪声，第 10 个 epoch 出现天空、地面和物体色块，第 20 个 epoch 能看出车、动物和船的大致形状，细节仍然模糊。

固定噪声的生成过程（`runs/cifar10_gan/samples_progress.png`）：

<img width="1330" height="476" alt="image" src="https://github.com/user-attachments/assets/9995236c-bcab-40f1-81e9-32b3e0a1d9e3" />


训练曲线（`runs/cifar10_gan/loss.png`）：

<img width="1330" height="476" alt="image" src="https://github.com/user-attachments/assets/65b2da0f-c5b0-4e28-8410-d2cd879f615b" />


测试集上的判别器打分，以及真实测试图与生成图对照（`runs/cifar10_gan/test_eval.png`）：

<img width="1764" height="588" alt="image" src="https://github.com/user-attachments/assets/0754dabc-ff9a-4b3f-a7ef-3dbf648e31d6" />


---

## 简介

生成对抗网络（Generative Adversarial Network，简称 GAN）由 **Ian Goodfellow** 等人于 2014 年提出（NeurIPS 2014）。它把生成模型写成一场两人博弈：生成器 G 把噪声 z 映射成样本，判别器 D 判断一张图来自数据集还是来自 G。D 尽量把真实图像打高分、把生成图像打低分；G 尽量让自己的输出也被打成高分。两边都足够强、并且生成分布已经和数据分布一致时，D 对真实图和生成图都输出 0.5。

论文第 3 节指出，如果让 G 直接去压低「D 把假样本判成假」的那一项，训练早期梯度很弱，因为那时 D 很容易把假样本打成接近 0。实际训练改成让 G 去抬高 D(G(z))，也就是 **non-saturating** 生成器目标。这个目标和原来的博弈在同一点达到均衡，早期梯度更可用。

GAN 之前的深度生成模型多半要写一个显式密度，再用变分下界或马尔可夫链去近似。GAN 的判别器提供一个可学习的损失，生成器只要把噪声前向映射成图像，采样时不需要马尔可夫链。

本复现用 PyTorch 写出这场博弈。判别器一步、生成器一步交替更新，对应论文 Algorithm 1 里 k = 1 的设定。生成器使用 non-saturating 目标。两个实验分别是 MNIST 上的全连接网络，以及 CIFAR-10 上的卷积网络。评估看两边的分数是否向 0.5 靠近，以及固定噪声下的图像是否从噪声变成可辨认的样本。

## 架构

整体是生成器和判别器。噪声先进入 G，G 的输出和真实图像一起进入 D。D 在本实现里输出 logit，Sigmoid 放在 `BCEWithLogitsLoss` 内部，和先 Sigmoid 再算二元交叉熵是同一个目标，数值上更稳。

<img width="1280" height="546" alt="image" src="https://github.com/user-attachments/assets/953089ed-b4da-4705-9780-dd1d1539b708" />


- **生成器 G**：把 100 维标准正态噪声映射成一张图。MNIST 上是三层全连接，宽度 256、512、1024，隐层 ReLU，输出层 Sigmoid，得到范围在 0 到 1 的 784 维向量。CIFAR-10 上是四层转置卷积，通道 256、128、64、3，隐层接 BatchNorm 和 ReLU，输出层 Tanh，得到范围在 -1 到 1 的 3×32×32 图像。
- **判别器 D**：判断输入是真实图像还是 G(z)。MNIST 上是三层全连接，宽度 1024、512、256，每层 LeakyReLU(0.2) 和 Dropout(0.3)，最后一层线性输出 logit。CIFAR-10 上是四层卷积，通道 64、128、256、1，除第一层外接 BatchNorm，隐层 LeakyReLU(0.2)，最后一层输出一个 logit。
- **判别器损失**：真实图像的目标是 1，生成图像的目标是 0。它对应「把真样本判真、把假样本判假」。
- **生成器损失**：生成图像的目标是 1。它对应 non-saturating 目标，也就是让 D 把假样本判成真。
- **R1 惩罚**：在真实图像上惩罚 D 对输入的梯度大小。这一项来自 Mescheder 等人 2018 年的收敛性分析，用来把判别器在真实样本附近压平，避免分数贴在 0 和 1 上震荡。MNIST 的系数是 10，CIFAR-10 的系数是 1。曲线里的 D loss 含有这一项。均衡点仍然是 D = 0.5。

了解更多架构可以见文章：[GAN深入浅出](https://blog.csdn.net/Eric005/article/details/171045540?fromshare=blogdetail&sharetype=blogdetail&sharerId=171045540&sharerefer=PC&sharesource=Eric005&sharefrom=from_link)

**和 2014 年论文对齐的部分**是这场最小最大博弈、Algorithm 1 的交替更新（k = 1），以及第 3 节的 non-saturating 生成器。本实现同时做了几处后来成为常规做法的改动：优化器使用 Adam（学习率 2e-4，beta 为 0.5 和 0.999），判别器隐层使用 LeakyReLU 和 Dropout，噪声先验是标准正态。2014 年论文的判别器使用 maxout，优化器是带动量的小批量 SGD。CIFAR-10 的卷积结构来自 DCGAN（Radford 等人，2016）：转置卷积逐级放大，卷积权重用标准差 0.02 的正态初始化，BatchNorm 的缩放初始化为均值 1、标准差 0.02 的正态。

| | MNIST 全连接 | CIFAR-10 卷积 |
| --- | --- | --- |
| 生成器参数量 | 1,486,352 | 1,069,827 |
| 判别器参数量 | 1,460,225 | 664,066 |
| 噪声维度 | 100 | 100 |
| 输出 | 784 维，Sigmoid，0 到 1 | 3×32×32，Tanh，-1 到 1 |
| R1 系数 | 10 | 1 |
| batch / epoch | 128 / 20 | 128 / 20 |

## 数据集

两个实验共用上面的对抗目标和 k = 1 的交替更新。数据、网络和像素范围不同。都不做翻转或裁剪。

### 实战 1：MNIST

MNIST 是 28×28 的单通道手写数字，类别 0 到 9。

- **规模**：训练集 60000 张，测试集 10000 张。
- **预处理**：只做 `ToTensor()`，像素落到 0 到 1，与生成器的 Sigmoid 输出一致。
- **训练**：batch 128，学习率 2e-4，Adam，最多 20 个 epoch，种子 0。每步先更新一次判别器，再更新一次生成器。同一批 64 个噪声从第 1 个 epoch 保留到最后，用来看生成结果的变化。
- **指标**：训练中记录 D loss、G loss、D(real) 和 D(fake)。测试时在官方 test split 上统计判别器分数。价值函数是「真实图被判真的对数概率」加上「生成图被判假的对数概率」，均衡时约为 -1.386。

这一轮结束时，训练集上 D(real) = 0.503、D(fake) = 0.497，D loss = 1.381，G loss = 0.702。测试集直方图里，真实图和生成图的分数都挤在 0.5 附近。

| 判别器分数 | 效果说明 |
| --- | --- |
| 一边接近 1、一边接近 0 | 判别器还能轻松分开真假，生成器尚未跟上 |
| 两边向 0.5 移动，G loss 往 0.69 降 | 分布开始接近，博弈进入均衡附近 |
| D(real) 和 D(fake) 都约等于 0.5，D loss 约 1.386，G loss 约 0.693 | 纳什均衡附近。MNIST 这一轮停在这里 |
| 分数长期贴在 0 或 1，损失剧烈震荡 | 判别器过强或训练发散，生成图像会糊成少数模式 |

数据集官方地址：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### 实战 2：CIFAR-10

CIFAR-10 是 32×32 的 RGB 图像，10 类：airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。2014 年 GAN 论文也在这个数据集上给出了样本。本实验用卷积生成器和判别器保留空间结构。

- **规模**：训练集 50000 张，测试集 10000 张。
- **预处理**：`ToTensor()` 之后按均值 0.5、标准差 0.5 归一化，像素落到 -1 到 1，与 Tanh 输出一致。显示时再线性映射回 0 到 1。
- **训练**：batch、学习率、优化器和 epoch 数与 MNIST 相同。R1 系数改为 1。卷积和 BatchNorm 使用 DCGAN 的正态初始化。
- **指标**：同样记录两侧损失和判别器分数，并在 test split 上画出分数直方图。上四行是真实测试图，下四行是 G(z)。

这一轮结束时，D(real) = 0.545、D(fake) = 0.455，D loss = 1.305，G loss = 0.854。两边还在靠拢。20 个 epoch 的样本已经有颜色和粗略物体形状，笔画级的细节还没有出现。这个数字衡量的是短程训练后的均衡程度，不能和训练了更多 epoch 的 DCGAN 样本直接比较。

数据集官方地址：[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Introduction

The Generative Adversarial Network (GAN), introduced by **Ian Goodfellow** and coauthors at NeurIPS 2014, trains two networks against each other. The generator G maps noise z to a sample. The discriminator D says whether an image came from the dataset or from G. D tries to score real images high and generated images low. G tries to make its own outputs score high as well. When both networks are strong and the generated distribution matches the data, D outputs 0.5 for real and generated images alike.

Section 3 of the paper notes that pushing down the term "D correctly calls a fake sample fake" gives a weak gradient early in training, because D rejects fake samples with scores near 0. The practical generator objective instead pushes D(G(z)) up. This is the **non-saturating** loss. It shares the same equilibrium and supplies a usable gradient while G is still poor.

Earlier deep generative models usually wrote an explicit density and approximated it with a variational bound or a Markov chain. A GAN discriminator is a learned loss. Sampling is one forward pass of G.

This reproduction implements that game in PyTorch. One discriminator step alternates with one generator step, which is Algorithm 1 with k = 1. The generator uses the non-saturating objective. The two runs are a fully connected GAN on MNIST and a convolutional GAN on CIFAR-10. The check is whether both scores move toward 0.5, and whether images drawn from a fixed noise batch change from noise into recognizable samples.

## Architecture

A noise vector enters G. Real images and G's outputs enter D. In this code D returns a logit. The sigmoid sits inside `BCEWithLogitsLoss`. That is the same objective as a sigmoid followed by binary cross-entropy, with a more stable backward pass.

<img width="2223" height="1093" alt="image" src="https://github.com/user-attachments/assets/844979bf-bb99-487e-bbdb-b861ab831db2" />


- **Generator G**: Maps 100-dimensional standard normal noise to an image. On MNIST it is a three-layer MLP of widths 256, 512, and 1024, with ReLU hidden units and a Sigmoid output, a 784-dimensional vector in 0 to 1. On CIFAR-10 it is four transposed convolutions with channel counts 256, 128, 64, and 3. Hidden layers use BatchNorm and ReLU. The output is Tanh, a 3×32×32 image in -1 to 1.
- **Discriminator D**: Scores an image as real or generated. On MNIST it is an MLP of widths 1024, 512, and 256, each with LeakyReLU(0.2) and Dropout(0.3), then a linear logit. On CIFAR-10 it is four convolutions with channel counts 64, 128, 256, and 1. BatchNorm is applied after every convolution except the first. Hidden layers use LeakyReLU(0.2). The last layer is a single logit.
- **Discriminator loss**: Target 1 on real images and target 0 on generated images. This is "call real samples real, and fake samples fake."
- **Generator loss**: Target 1 on generated images. This is the non-saturating objective: make D call fake samples real.
- **R1 penalty**: Penalizes the size of D's gradient with respect to real images. This term is from Mescheder et al. (2018). It flattens D near the data so the scores do not stick to 0 and 1. MNIST uses coefficient 10 and CIFAR-10 uses coefficient 1. The plotted D loss includes this term. The equilibrium of the scores is still D = 0.5.

**What matches the 2014 paper** is the minimax game, the alternating update in Algorithm 1 with k = 1, and the non-saturating generator in Section 3. This code also uses choices that became standard later. The optimizer is Adam with learning rate 2e-4 and betas 0.5 and 0.999. The discriminator hidden layers are LeakyReLU and Dropout. The noise prior is a standard normal. The 2014 discriminator used maxout units, and the paper optimized with minibatch SGD with momentum. The CIFAR-10 networks follow DCGAN (Radford et al., 2016): transposed convolutions that grow the spatial size, convolutional weights initialized from a normal with standard deviation 0.02, and BatchNorm scales initialized from a normal with mean 1 and standard deviation 0.02.

| | MNIST MLP | CIFAR-10 conv |
| --- | --- | --- |
| Generator parameters | 1,486,352 | 1,069,827 |
| Discriminator parameters | 1,460,225 | 664,066 |
| Noise dimension | 100 | 100 |
| Output | 784, Sigmoid, 0 to 1 | 3×32×32, Tanh, -1 to 1 |
| R1 coefficient | 10 | 1 |
| Batch / epochs | 128 / 20 | 128 / 20 |

## Dataset

Both runs share the adversarial objective and the k = 1 update. The data, the networks, and the pixel range change. Neither run flips or crops images.

### Experiment 1: MNIST

MNIST is 28×28 grayscale handwritten digits, classes 0 through 9.

- **Scale**: 60,000 training images and 10,000 test images.
- **Preprocessing**: `ToTensor()` only, so pixels lie in 0 to 1, matching the Sigmoid output.
- **Training**: Batch size 128, learning rate 2e-4, Adam, 20 epochs, seed 0. Each step updates the discriminator once and then the generator once. The same 64 noise vectors are kept from epoch 1 so the grids show how one batch of z changes.
- **Metrics**: D loss, G loss, D(real), and D(fake) during training. At test time the discriminator scores the official test split. The paper's value is the log probability of calling real images real, plus the log probability of calling generated images fake. At equilibrium it is about -1.386.

At the end of this run, D(real) = 0.503, D(fake) = 0.497, D loss = 1.381, and G loss = 0.702 on the training set. On the test histogram, real and generated scores both sit on 0.5.

| Discriminator scores | Reading |
| --- | --- |
| One side near 1 and the other near 0 | D still separates real from fake easily, and G has not caught up |
| Both sides moving toward 0.5, G loss falling toward 0.69 | The two distributions are getting closer |
| D(real) and D(fake) both near 0.5, D loss near 1.386, G loss near 0.693 | Near the Nash equilibrium. The MNIST run stops here |
| Scores stuck at 0 or 1, with a swinging loss | D is too strong or training is diverging, and samples collapse onto a few modes |

Official dataset page: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### Experiment 2: CIFAR-10

CIFAR-10 is 32×32 RGB images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The 2014 paper also shows samples on this dataset. This run uses convolutional G and D so the spatial layout is kept.

- **Scale**: 50,000 training images and 10,000 test images.
- **Preprocessing**: `ToTensor()`, then normalization with mean 0.5 and standard deviation 0.5, so pixels lie in -1 to 1 and match the Tanh output. Grids are mapped back to 0 to 1 for display.
- **Training**: The batch size, learning rate, optimizer, and epoch count match the MNIST run. The R1 coefficient is 1. Convolutions and BatchNorm use the DCGAN normal initialization.
- **Metrics**: The same losses and scores, plus a score histogram on the test split. The top four rows are real test images and the bottom four rows are G(z).

At the end of this run, D(real) = 0.545, D(fake) = 0.455, D loss = 1.305, and G loss = 0.854. The two scores are still closing. After 20 epochs the samples have color and coarse object shapes. Fine detail is still missing. These numbers describe a short training run. They are not a comparison against a DCGAN trained for many more epochs.

Official dataset page: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 原文章 | Original article

Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, Courville A, Bengio Y. Generative Adversarial Nets[C]//Advances in Neural Information Processing Systems. 2014: 2672-2680.

arXiv: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

本复现额外用到的两篇后续工作：

Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[C]//International Conference on Learning Representations (ICLR). 2016. arXiv: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)

Mescheder L, Geiger A, Nowozin S. Which Training Methods for GANs do actually Converge?[C]//International Conference on Machine Learning (ICML). 2018: 3481-3490. arXiv: [https://arxiv.org/abs/1801.04406](https://arxiv.org/abs/1801.04406)
