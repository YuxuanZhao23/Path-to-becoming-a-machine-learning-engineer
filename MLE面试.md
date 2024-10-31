# Introduction

ML algorithm 不是全部，还有很多重要的 component 保障了模型的高效运作

- data collection/ verification
- feature extraction
- monitor
- evaluate
- analysis
- process/ resource management
- service infrastructure

所有的 SD 问题都可以有以下的步骤

## 明确要求

做好笔记：
- business objective: improve revenue? visits?
- 需要什么 feature
- data：数据源是什么，有多大，是否有label
- constraint：有多少 computing power？云还是on device？模型会不会随着时间变好？
- scale：有多少用户？处理多少数据？有哪些指标需要增长多少？
- performance：多快？real time？priority：accuracy vs latency
- privacy & ethics

## 将问题定义为 ML 任务

### 定义 ML 的目标

什么是好的目标？不同的目标怎么比较？

e.g.
- Maximize the number of event registrations
- Maximize the time users spend watching videos
- Maximize click-through rate
- Accurately predict if a given content is harmful
- Maximize the number of formed connections

### 指定输入和输出

可能需要多个模型，或者一个模型有多个输出

我们可以有一个模型 predict violence，另一个 predict nudity

### 选择正确的 ML category

- supervised learning
  - classification
    - binary
    - multiclass
  - regression：有输出的范围吗
- unsupervised learning
- reinforcement learning

## 数据准备

### data engineering

- source: who collect it? clean? trusted? UGC or system generate? 数据量有多大，新数据的出现频率
- storage: SQL/ NoSQL，在云端还是用户设备上？
- ETL Extract source, Transform to a format, Load to file/ database
- Data Types
  - structured
    - numerical
      - discrete
      - continous
    - categorical
      - ordinal：开心，中立，不开心可以算作有连续顺序
      - nominal: 没有数字/先后顺序的，比如性别
  - unstructured
    - audio
    - video
    - image
    - text

### feature engineering

- 使用 domain knowledge 从原数据中 extract features
- missing
  - deletion: 缺的多就删除整列，缺的少就删除行
  - imputation：使用默认值/平均值/中位数/众数mode
- scaling
  - normalization: min-max $z = \frac{x - x_{min}}{x_{max} - x_{min}}$
  - standardization (z-score normalization) $z = \frac{x - \mu}{\sigma}$ $\mu$ mean, $\sigma$ standard deviation
  - log: 使得数据的 distribution less skewed
- discretization: bucketing 将 continuous 特征变成 categorical
- encoding categorical
  - integer：如果分类具有数字的 natural relationship
  - one-hot: 如果没有数字关系的时候
  - embedding learning：如果有很多很多的类的时候，使用 unique n-dimensional vector 来代表

### privacy & ethics

有多 sensitive？用户是否担心？是否需要 anonymization？是否可以将数据存在我们的服务器上？还是只能在他们的设备上访问？

bias：存在吗？如何纠正？

## 模型开发

### 模型选择

- 使用简单的模型来测试一下，比如 logistic regression
- 尝试使用多种复杂的模型，然后使用 ensemble
  - bagging
  - boosting
  - stacking
- 一些比较典型的模型
  - Logistic regression
  - Linear regression
  - Decision trees
  - Gradient boosted decision trees and random forests
  - Support vector machines
  - Naive Bayes
  - Factorization Machines (FM)
  - Neural networks
- 模型训练的时间/训练数据/computing resource
- 模型 inference 的 latency
- on device/ cloud
- 多少 parameter，需要多少 memory
- 如何选择超参数？hidden layer/neurons/activation的数量

### 模型训练

#### 构建 dataset

1. collect raw data：前面说了
2. identify feature + label：feature 前面说了，label 可以是 hand 或者是 natural，natural 也就是说我们在网站中收集的比如说点赞或者点踩
3. sampling：如果我们只收集一部分的数据，那么我们可以使用这些常见的方法：convenience sampling, snowball sampling, stratified sampling, reservoir sampling, and importance sampling
4. split data：把数据分成 training, evaluation/ validation, test
5. address class imbalance：可以 resample (oversample/ downsample)，也可以在 loss function 给 minority 更多的权重

#### 选择 loss function

一般是在已有的 loss function里选择，或者进行细微的调整：
- Cross Entropy
- MSE
- MAE
- Huber loss

regularization：
- L1/ L2
- Entropy regularization
- K-fold CV
- dropout

什么是 back propagation?

optimization：
- SGD
- AdaGrad
- Momentum
- RMSProp

Activation：
- ELU
- ReLU
- Tanh
- Sigmoid

overfit 和 underfit 的可能原因是什么？如何处理？

#### from scratch 还是 fine tunning

design choice

#### distributed training

- data parallelism: limited by batch size
- model parallelism: cut the model by conv layer, FC

## 评估

### offline evaluation

| Task | Offline metrics |
| ---- | --------------- |
| Classification | Precision, recall, F1 score, accuracy, ROC-AUC, PR-AUC, confusion matrix|
|Regression | MSE, MAE, RMSE |
| Ranking | Precision@k, recall@k, MRR, mAP, nDCG |
| Image generation | FID, Inception score |
| Natural language processing | BLEU, METEOR, ROUGE, CIDEr, SPICE |

### online evaluation

选择 online evaluation metrics 一般是非常主观的，取决于 business sense

| Problem | Online metrics |
| ------- | -------------- |
| Ad click prediction | Click-through rate, revenue lift, etc. |
| Harmful content detection | Prevalence, valid appeals, etc. |
| Video recommendation | Click-through rate, total watch time, number of completed videos, etc. |
| Friend recommendation | Number of requests sent per day, number of requests accepted per day, etc. |

## 部署和服务

### cloud/ on-device

cloud: 简单，inference一般更快，限制比较小

on-device：便宜，没有网络延迟（可以不联网），更多的privacy

### model compression

- knowledge distillation：用一个更大的模型教一个小模型
- pruning：将不重要的参数设为0，之后可以用sparse matrix来储存节省空间
- quantization：使用16位来训练

### test in production

- shadow deployment: 新旧模型都在被使用，但是我们给用户返回的是旧模型的结果。成本较高
- A/B testing：并行部署，一部分的 traffic 被随机路由到新的模型，都有足够的data point
- canary release
- interleaving experiments
- bandits

### prediction pipeline

- batch prediction: 定期的，成本低，预先计算需要知道用户的输入，不需要实时出结果也行（queue）
- online prediction：用户请求的时候才进行

## 监控和基础设施

### 为什么 fail？

data distribution shift：prod 的数据和训练的不一样，性能可能会随着时间变差因为模型 stale

- 更大的数据集上训练，使得prod上面的data point也在这个distribution上
- 定期在新的distribution 上进行 retrain

### monitor 什么？

- operation metrics
  - average serving times
  - throughput
  - the number of prediction requests
  - CPU/GPU utilization
- ML metrics
  - I/O
  - I/O distribution drifts
  - accuracy
  - version