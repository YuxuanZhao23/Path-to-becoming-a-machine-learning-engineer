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

# YouTube 视频推荐

- Can I assume the business objective of building a video recommendation system is to increase user engagement?
- Does the system recommend similar videos to a video a user is watching right now? Or does it show a personalized list of videos on the user’s homepage?
- can I assume users are located worldwide and videos are in different languages?
- Can I assume we can construct the dataset based on user interactions with video content?
- Can a user group videos together by creating playlists? Playlists can be informative for the ML model during the learning phase.
- How many videos are available on the platform? 10 billion videos
- How fast should the system recommend videos to a user? Can I assume the recommendation should not take more than 200 milliseconds?

## 定义 ML 目标：

- 最大化 click：可能会推荐 clickbait 视频，最终减少了 user satisfaction & engagement
- 最大化完播率 completed videos
- 最大化 total watch time
- 最大化 relevant videos：定义 relevant 是自由定义的，可以是 like 或者 watch at least half of it

## 输入和输出：

输入是用户的行为和视频，输出是视频的 relevance scores

## ML category

### Content-based filtering

用户喜欢视频 X,Y + Z 和视频 X,Y 非常类似 => 推荐 Z

- Pros: 可以给 unique interest 推荐
- Cons: 很难 discover new interest，需要 domain knowledge

### Collaborative filtering

给用户 cluster，然后我们假定相似的用户对相似的视频感兴趣

- pros: 不需要 domain knowledge 也很高效，因为我们不依赖 video feature。容易 discover users's new interest
- cons: cold-start 缺乏用户和视频信息的时候没办法进行准确的推荐。不能 handle niche interest，因为没有什么 similar users

### Hybrid filtering

同时使用 content based 和 CF based

## Data Preparation

### Data Engineering

- Videos: 视频本身和 metadata：id, length, title ...
- Users: id, name, age, gender, city, country, language, time zone ...
- Interaction: user id, video id, interaction type, interaction value, location, timestamp

### Feature Engineering

Video Feature

- video id: categorical embedding as numerical vector
- duration: numerical
- language: categorical embedding as numerical vector
- tag: light-weight model 像 CBOW 来 map feature vectors，然后 aggregate
- title: BERT => embedding

User Feature

- User demographics: embedding all id, name, age, gender, city, country, language, time zone ...
- Contextual information: time of day, day of week, device
- User historical interactions
  - search history: interest，过去的行为往往是未来行为的 indicator
  - liked video: interest
  - watched videos

## Model Development

### Feedback Matrix

1 代表 observed 或者 positive

|user \ video | 1 | 2 | 3 | 4 |
| ----------- | - | - | - | - |
| 1 | 1 | 1 |  |  |
| 2 |  | 1 | 1 |  |
| 3 |  | 1 |  | 1 |

- Explicit feedback: like/ share
- Implicit feedback: click/ watch time
- Combination

我们的 ML 目标是最大化 relevancy，而 relevancy 可以被定义成 combination

#### matrix factorization

把 feedback matrix 分拆成两个小矩阵的乘积，分别代表 user embedding 和 video embedding，那我们的训练目标就是训练这两个 embedding 使得他们的乘积尽可能与事实相似

随机初始化这两个矩阵，然后优化 loss，可以有两种 loss

1. sqaured distance of observed <user, video>: $loss = \displaystyle\sum_{(i, j)\in obs}(A_{ij} - U_i U_j)^2$
2. sqaured distance of <user, video>: $loss = \displaystyle\sum_{(i, j)}(A_{ij} - U_i U_j)^2$ 这个方法不好因为 feedback matrix 一般是稀疏的，如果惩罚未观察到的位置就会使得绝大多数的预测接近于0，对未见过的用户和视频 generalize 能力很差
3. weighted: $loss = \displaystyle\sum_{(i, j)\in obs}(A_{ij} - U_i U_j)^2 + W\displaystyle\sum_{(i, j)\not\in obs}(A_{ij} - U_i U_j)^2$ 避免其中一个 dominate，实际中使用这一个

#### optimization

- SGD
- Weighted Alternating Least Square (WALS) 特别适合矩阵分解：fix one optimize another, vice versa

#### inference

对于任意的视频和用户，都可以点积得到一个relevance score

#### 优缺点

- pros: 训练速度很快，serving 速度很快
- cons: 只依赖用户和视频的交互，没有使用其他的 feature 比如 age/ language。很难给新用户推荐，因为没有足够的 interaction 来给用户构建 embedding

### Two Tower Neural Network

User feature => User Encoder (DNN) => User Embedding

Video feature/ Video id => Video Encoder (DNN) => Video Embedding

两个 embedding 的distance 就是他们的 relevance

#### 构建数据集

用户 like 或者观看了超过一半的视频，我们标记成 positive，而用户 dislike 和不相关的随机视频标记成 negative，negative 肯定会更多，所以我们使用之前提到的 resample (oversample/ downsample)，也可以在 loss function 给 minority 更多的权重

#### 选择损失函数

因为是 binary labels，所以我们可以当成 classification 来做，使用典型的 cross entropy 就可以

#### 优缺点

- pros: 使用 user feature，很容易 handle 新用户，效果更好（因为用到了更多的用户信息）
- cons: 训练很贵，serving 很慢（因为用户的 feature embedding 需要在查询的时候计算）

## Evaluation

Offline Metrics

- precision@k: top k recommend video 的相关视频数量 (k = 1, 5, 10 ...)
- mAP: ranking quality
- diversity: 推荐的 list 里面的 average pairwise similarity 应该小一点，因为用户喜欢更多的 diversity，但是应当结合起来看，多样性可能会损害 relevancy

Online Metrics

- CTR: 很有用的指标，但是需要小心 clickbait video
- completed video：推荐的视频用户真的喜欢吗
- watch time：假设推荐引起了用户的兴趣，那么他们会花更多的时间看视频
- explicit feedback：like/ dislike

## Serving

我们有很多视频，如果使用一个效果很好但是速度比较慢的模型，会使得我们的serving受到影响，所以我们可以用轻量级的模型先快速缩小范围 (candidate generation)，然后再使用更重的模型进行评分和排名 (scoring => reranking)

### Candidate Generation

轻量级：要很快，应该可以处理新用户，不依赖视频特征，可以用 two tower NN 来 approximate nearest neighbor 缩小到千的量级（不怕false positive）

现实中会使用多个 candidate generation，来提供不同需求的候选（增加 diversity）：relevant, popular, trending

### Scoring

仍然使用 two tower nn，但是我们需要视频的feature了

### reranking

- 可以使用独立的 ML 模型来分辨是否是 clickbait
- 视频有没有 region restrict
- freshness 时效性
- misinformation
- duplicate/ near duplicate
- fair/ bias

## 主要的挑战

新用户：使用用户的 feature 来推荐 (age, gender, language, location)

新视频：只有 metadata 和 video 本身，没有 interaction，我们可以给随机用户展示并收集 interaction

使用神经网络来便于 continuously fine-tuning

# News Feed

- keep users engaged with the platform?
- activity consists of both unseen posts and posts with unseen comments?
- post contain textual content, images, video, or any combination?
- the system should place the most engaging content at the top of timelines, as people are more likely to interact with the first few posts. Does that sound right?
- Is there a specific type of engagement we are optimizing for? I assume there are different types of engagement, such as clicks, likes, and shares.
- What are the major reactions available on the platform? I assume users can click, like, share, comment, hide, block another user, and send connection requests. Are there other reactions we should consider?
- run < 200ms
- daily active users: We have almost 3 billion users in total. Around 2 billion are daily active users who check their feeds twice a day.

## ML task

- 最大化 implicit reaction：dwell time/ click 数据比较多，但是用户可能点进去之后发现不值得看（数据有误导性）
- 最大化 explicit reaction: like, share, hide 通常 more weighted，但是用户很少作出这样的反应
- weighted both

## I/O

某个特定用户是输入，输出是一系列 sorted post

## 选择 ML category

Pointwise Learning (LTR) 根据 engagement score 来排序帖子：我们给某种行为赋予一个 value，然后预测这种行为出现的概率相乘之后全部相加得到 score

## Data Preparation

User: id, name, age, gender, city, country, language, time zone

Post: content, hashtag, mention, image/ video, timestamp

User-Post Interaction: user id, post id, interaction type, interaction value, location, timestamp

Friendship: user1, user2, timestamp when formed, close friend, family

## Feature Engineering

### Post Feature

- text: 内容，用 BERT 做 embedding
- image/ video：ResNet 或者 CLIP 做 embedding
- reaction：like, share, comment 体现了有多吸引人，scale (log) 来限制在 similar range
- hashtag：需要 tokenize，然后转换成 id（使用 feature hashing），因为不需要上下文所以不用 Transformer 反而使用 lightweight 的 TF-IDF or word2vec
- post's age：用户倾向于 interact 更新的内容

### User Feature

- Demographics: age, gender, country, etc
- Contextual information: device, time of the day, etc
- User-post historical interactions：过去可能反映将来
- Being mentioned in the post：用户可能会更关注被提到的 post，用1/0表示

### User-author affinities

affinities 亲和力，friendship length

Like/click/comment/share rate：这个用户给另一个用户 95% 的帖子都点赞了，可能他们是很好的朋友或者家人

## Model Development

我们想要使用 deep neural network

- 可以很好地处理 unstructured data
- 允许我们使用 embedding 来代表 categorical features
- 可以 fine-tuning pretrained model

### 选择模型

- n 个独立的 DNNs：训练起来很贵，而且有的 reaction 太少没有足够的 training data
- Multi-task DNN：增加两个 feature 来预测 passive users：dwell time 和 skip（在这个帖子上只停留了1s）

### 模型训练

构建数据集，避免负面 data point 太多，平衡一下

dwell time 是一个 regression task

### loss function

所有的loss需要加起来，binary classification 使用 binary cross entropy，regression loss 使用 MAE, MSE, Huber loss

## Evaluation

offline：
- classification 可以使用 recall 和 precision 来判断
- 使用 ROC curve 来了解 true positive rate 和 false positive rate
- 计算 ROC curve 下的面积 (ROC-AUC) 来用数字总结 binary classification 的表现

online：
- CTR = $\frac{|click|}{|impression|}$
- reaction rate = $\frac{|like|}{|impression|}$
- total time spent: 比如说一周内使用的时间
- user satisfaction rate：使用 user survey

## Serving

prediction pipeline：

- retrieval service：还没阅读的
- ranking service：分配 engagement score 来排名
- reranking：根据额外的 logic 和 filter 来推送用户更有可能喜欢的内容（比如说用户明确表示喜欢足球）