# unclip/ DALL-E 2

文章主要的 novelty 是 Stable Diffusion

DALL-E 有一个层级式的结构，不断上采样生成更清晰的图片

1. 给定一个文本，然后用一个 freezed CLIP 模型把它转化成 text embedding
2. 用 prior 的方法从 text embedding + text本身 转化成 image embedding（CLIP 生成的 image embedding 作为 ground truth 在这里）这个方法的主要好处是对比 GAN 生成的图片更有多样性，这个 prior 是一个扩散模型
3. 然后用 decoder 把这个 embedding 转换成图片，这个 decoder 是一个扩散模型

背景知识

GAN 是训练一个 generator 和一个 discriminator，所以生成图片的保真度非常高，人眼很难分辨区别。缺点是多样性比较差以及比较难同时训练两个模型

Auto Encoder：encoder 把图像变得很窄的embedding(bottleneck)，然后再 decode 回去同一张图片
- 一个比较好的改进是将原始输入 denoised 或者 masked，因为图像的信息冗余很多
- VAE 中间生成的不是 embedding，而是一个高斯分布，学会了这个高斯分布之后其实前面的 encoder 就不再需要。我们每次在这个高斯分布里面随机取样 z，prior + decoder 生成图片，最大化 $p(x'|z)$
- VQVAE 其实现实中的数据都是离散的，所以计算一个分布不如直接用一个 8192 x 512/768 的 codebook 记下来表现力强，也就是说我们有 8192 个聚类中心

DALL-E 2 模型

- text 用 Byte Pair Encoding (BERT, GPT) 抽成文本的 256 feature
- image 256 x 256 通过 VQVAE 抽成图像的 32 x 32 feature
- concatenate 在一起之后输入到 12B 的 GPT 中
- 会生成很多张图片，用CLIP给这些图片排名找到最 relevant
- decoder 只用了 UNet 这种纯CNN的方案，里面没有 transformer
- diffusion prior 里面用了 transformer decoder，但是发现预测噪声没用，还是每一步直接预测图片
- 因为用了 CLIP 所以更多是物体的出现，对于语义上的理解还有偏差

Diffusion

给一个图片逐渐加入噪音直到变成完全的噪音分布，这个过程被称为 forward diffusion。把这个过程反过来就是，给定噪音，返回一张清晰合理的图片，这个过程被称为 reverse diffusion。一般这个过程是100步，所以训练起来很贵，inference也很贵

那么每一步是在做什么呢：UNet 是一个类似于沙漏的设计，由大到小的 CNN 之后接着由小到大的 CNN。往往会有一些 skip connection 让图像恢复得更好

DDPM

- 从 $x_t$ 预测 $x_{t-1}$ 太难了，那我们学一下噪音 $\epsilon$，这其实是类似于 resnet 的思想
- 会给每一步加上一个time embedding（类似于transformer里面的位置编码，是正弦/傅里叶特征），告诉模型现在训练到第几步了
- 因为每一步的 UNet 都是共享参数的，所以如何控制模型一开始关注于大体和轮廓，最后关注细节和高频特征呢？就是用提醒模型现在的时间来做到的
- 因为噪音都是主动加的，所以每一步的loss就是loss 和预测的loss之间的差异
- DDPM 只需要知道均值，不需要知道方差，只要 fix 一个方差就好了

improved DDPM (OpenAI)

- sin -> cos
- 把方差学了之后效果更好了

classifier guided Diffusion

在每一步都用 imagenet 这样的数据集训练出来的 classifier 去分类，这样就可以计算出来梯度帮助模型训练。因为如果 classifier 能识别出来某种比较强的物体信号，那么证明图像生成得很成功（很逼真，能够和 GAN 匹配的逼真程度）DALL-E 2 用的是一个 clip guide

classifier free guided（效果比较好）

在 forward 的过程生成两个结果，一个是有接收文本输入的，一个是没有的。然后去比较两者间的差异。但是这样很贵，因为 diffusion 本身就很贵，forward 两次是不可以接受的。OpenAI也同时用了这个 guided，会随机扔掉一些 embedding，text在一半的时间内也 drop 了