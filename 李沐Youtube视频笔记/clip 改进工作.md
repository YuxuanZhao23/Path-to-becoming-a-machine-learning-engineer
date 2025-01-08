# clip

将图片和对应的文本信息用双塔模型 encode 之后做正负样本的对比学习。之后拿到一张图片的时候，我们把潜在可能的类别作为 prompt 输入到 text encoder 然后找到最大概率是哪一个 prompt 就认为这张图片应当被分类成哪一类

# semantic segmentation Lseg

如果只看图像这一个塔的话，其实和有监督的segmentation是很类似的。

1. 首先使用了一个 DPT 的结构：ViT + decoder 把图像降维到 1/4 - 1/16 的同时channel增加到 512 或者 768
2. 然后这个feature map 做upscaling到一个图片相同大小的尺寸，
3. 最后模型的输出和ground truth supervision做一个cross entropy loss

文本塔的输入大小是可以随意改变的，比如说我们只检测狗，或者检测很多很多类都是可以的。经过 text encoder 之后会变成 n x c，n就是1或者很多类，c就是相同的 512 或者 768

将 h x w x c 和 n x c 相乘之后我们就得到了 h x w x n 的矩阵，这篇工作不是无监督的对比学习，而是用有标号的真实值来学习的。在实现中，即使有7个数据集，但是规模也只有几万十几万，如果fine-tuning clip 的 image encoder 和 text
encoder 很可能就训练歪了，所以作者锁住了 text encoder 完全没动，然后 image encoder 对比后发现 ViT 的模型参数要比 CliP 的表现要好

缺点：从实验数据来看，zero shot 和有监督的 segmentation 还是有差距的，换言之公司里面用 resnet 可能性能会好很多

# GroupViT

不依赖于手工标注的 segmentation（太贵了，数据量太小），而是用 text supervision

不用手工标注的话，那如何分辨哪些是同一个物体呢？作者使用了 grouping 的概念，也就是将像素逐渐合并在一起

模型架构

标准的 12 层 transformer

输入有两部分：
- 原始图像的 patch embedding：假设原图片 24 x 24，patch size 16 x 16，那么我们就有一个 14 x 14 x 196 长度的序列。本文是 196 x 384
- group tokens：作者设置为 64 x 384，64 是希望一开始有尽可能多的起始点/聚类中心，反正后面还可以合并
- 这里面的 group token 可以理解成 cls token，那为什么过去的图片 cls token 只有一个，但是这里却有多个呢？因为之前是一张图片分一个类别，但是现在是希望每一个小块都有一个单独的类别，这样才能做分割
- 在 6 个transformer block 之后，作者加入一个 grouping block，因为作者认为 clustering center已经学习得差不多了。这个 grouping block 的功能就是把之前的 patch embedding 直接 assign 到这 64 个 group tokens 上，同时 group token 也从 64 个降低到 8 个了
- grouping block 具体怎么操作的呢？用类似自注意力的方式先计算一个相似度矩阵，用这个矩阵帮助原来的 image token 做聚类中心的分配，从而将输入从 196 x 384 降维到 64 x 384。做聚类中心的过程是不可导的，所以这里用了一个小技巧 gumbel softmax
- 再 3 个 transformer block 之后，用 grouping block 进一步降维到 8 x 384
- 之后再接 3 层 transformer，但是文本这边只有 1 x 384，但是transformer这边输出是 8 x 384，作者用最简单的 average pooling 变成 1 x 384

缺点：
- 同样的 text zero shot 的结果和 supervised SOTA 还有较大差距
- 如何考虑背景类？不是一味地选择最大的相似度，因为有的时候最大的相似度也比较小。作者使用了阈值，就是说前景的embedding相似度要大于一个概率。但是对于类别很多的数据集，如果相似度设置的高了，那么所有的物体都被认定是背景。很低就会有错误分类
- 作者做了一个对比，发现其实segmentation已经做得很好，也就说说grouping很成功，只是预测label这一块的semantic segmentation没有做好。为什么？因为clip只能学非常清晰的分类，像背景这种语义非常模糊的label是学习不了的

# 目标检测 Vision and Language Knowledge Distillation ViLD

open vocabulary object detector：模型能够检测 novel category 的物体而无需新的标注

目标检测可以分成两个阶段，一个是 purpose bounding box，一个是 classifier。这篇文章只关注 classifier。

ViLD: 计算出来 N 个 embedding region 之后和 background 以及 Class base 的 text embedding 做点乘计算 cross entropy loss

ViLD-image：
- 把 proposal resize 之后输入给 Clip 训练好的 image encoder（这里 freeze weight），然后输出 M 个 image embedding
- 用 L1 loss 尽可能让ViLD 的输出 embedding 类似于 Clip 输出的结果，这样来达到 knowledge distillation 的目的
- 这个模型的 class 就不仅限于 class base 也可以有 class novel 了
- 这里面的 proposal 是 pre-computed，因为 clip 给出的 embedding 也是 pre-computed 的，不然速度太慢了
- 值得注意的是在inference的过程中，class base 和 class novel 都是明文具体的，这里并不涉及生成，只是计算 similarity

# Grounded Language-Image Pre-training GLIP

phase grounding 其实也是类似于目标检测的任务，给你一张图片和一句话，在图片里找到这句话里提到的东西

这个任务可以分为两个部分 detection 和 vision grounded

object detection classifier:
- Encode(Image) $\in R ^{N \times d}$
- Score = $OW^T \in R ^ {N \times c}$
- Ground Truth $\in \{0, 1\}^{N \times c}$
- Loss = Cross Entropy(Score, Ground Truth)

vision grounding 等同于 ViLD text 分支:
- O = Encode(Image)
- P = Encode(Prompt)
- Score = $OP^T$

两者间需要的连接是，什么情况下是 positive match

文章确定了这一点之后，用已经训练好的模型给没有标号的数据集加 pseudo label 然后用于训练。越多的数据训练的越好，最后在没有训练过的数据集上做 zero shot 也能有不错的 accuracy

# CLIPasso

用抽象的 stroke 来保留图片的 semantic 信息

Bezier Curve：用四个点控制的曲线，控制想要生成的简笔画的复杂程度是通过控制产生多少条 bezier curve 达到的。使用现成的一种 rasterizer（可导的） 转换成一副画

定义 loss
- Semantic Loss：使用 CLIP 来 encode 原画和简笔画应该都有相同的语义信息
- Geometric Loss：即使语义信息一致，画面仍有可能非常不同，用前期的特征维去算 loss（主要是保证物体的长宽，朝向等

基于 saliency 的初始化 Bezier Curve 的方式：saliency 其实是把输入放到 transformer 里面的最后一层的 self attention，然后再这张 saliency map 比较显著的地方去采集点 来生成 Bezier Curve

# Clip4Clip 视频 E2E retrieval

视频和图片最大的区别就是视频会有多个 embedding 的表示，那么如何做这个 many to one mapping 呢？

1. mean pooling：没有时序特征，下游数据不大的时候效果最好
2. transformer/ LSTM：sequential
3. transformer 同时接受文本和每一帧的 embedding，同时融合时序信息，视频帧，文本。下游数据集不大的情况下效果非常差，因为CLIP本身是在很大很好的数据上训练出来的，如果没有很好的数据去fine-tuning，其实不如直接freeze之后zero shot（不要画蛇添足）

# 动作识别 ActionCLIP

如果 batch 很大的话，那么同一个 batch 里面的一行就不一定是 one-hot 了（往往可能有多个图片都包含了相同的label），所以这里改成 KL Divergence 就好了

作者在视频这个塔加入了 pre/ in/ post- network prompt，其实是类似于 adapter 的概念，就是尽量利用现有的模型 freeze 住不动，然后只增加一些 plug-and-play 的小模块来直接处理下游任务
- pre: 给每一帧前面加上 temporal position 的信息
- in: shift 的概念，在每一个 ViT 之间增加一层 temporal shift module 来增强模型持续建模的能力
- pre 类似于 Clip4Clip

# 其他 empirical

Clip 用来做训练参数完成下游 vision-and-language tasks 会更好吗？会

在视频中用 CLIP + 语音，三种模态之间两两比较学习可以吗？可以

PointClip：把 3D 的 point cloud 投影到 2D 上之后可以用 Clip 吗？可以

用 prompt 的大小来预估物体在画面里的远近？Giant, close，等等