# ML task

business goal: user satisfaction/ wise investing decision

model goal: provide most relevant news feed

- 不是最大化 Click（避免 clickbait）
- 最大化 trade action，dwell time

# I/O

用户，行为，新闻 => 输出 relevance scores

# Collaborative filtering

cluster user，假定相似的用户会觉得相似的新闻有用

- pros: 不需要 domain knowledge 也很高效，因为我们不依赖 news feature，而是给用户推送已经奏效的 news
- cons: cold-start 缺乏用户和视频信息的时候没办法进行准确的推荐。不能 handle niche interest，因为没有什么 similar users（但是fidelity已经有很多用户的数据）

# Data Engineering

- News: text, id, length, title, timestamp, source
- Users: id, name, age, gender, city, country, time zone, portfolio
- Interaction: user id, news id, dwell time, timestamp, location

# Feature Engineering

## News Feature

- id: categorical => embedding
- length: numerical
- text, title: BERT => embedding
- timestamp: numerical
- source: categorical => embedding

## User Feature

- User demographics: embedding all id, name, age, gender, city, country, language, time zone ...
- Contextual information: time of day, day of week, device
- User historical interactions
  - search history: interest，过去的行为往往是未来行为的 indicator
  - watchlist, preious order: interest
  - previous read news

# Model

## Two Tower Neural Network

User feature => User Encoder (DNN) => User Embedding

Video feature/ Video id => Video Encoder (DNN) => Video Embedding

两个 embedding 的 distance 就是他们的 relevance

## 构建数据集

用户点击并停留的news，我们标记成 positive，而用户 skip（在这个news上只停留了1s） 和不相关的随机新闻标记成 negative，negative 肯定会更多，所以我们使用之前提到的 resample (oversample/ downsample)，也可以在 loss function 给 minority 更多的权重

## 选择损失函数

因为是 binary labels，所以我们可以当成 classification 来做，使用典型的 cross entropy 就可以

## Evaluation

Offline Metrics

- precision@k: top k recommend news 的相关数量 (k = 1, 5, 10 ...)
- mAP: ranking quality
- diversity: 推荐的 list 里面的 average pairwise similarity 应该小一点，因为用户喜欢更多的 diversity，但是应当结合起来看，多样性可能会损害 relevancy

Online Metrics

- CTR: 很有用的指标，但是需要小心 clickbait video
- dwell time：推荐的 news 用户真的喜欢吗
- total time spent：假设推荐引起了用户的兴趣，那么他们会花更多的时间使用这个网站
- explicit feedback：trade/ watchlist
- user satisfaction rate：使用 user survey

## Serving

我们有很多新闻，如果使用一个效果很好但是速度比较慢的模型，会使得我们的serving受到影响，所以我们可以用轻量级的模型先快速缩小范围 (candidate generation)，然后再使用更重的模型进行评分和排名 (scoring => reranking)

### Candidate Generation

轻量级：要很快，应该可以处理新用户，不依赖news特征，可以用 two tower NN 来 approximate nearest neighbor 缩小到千的量级（不怕false positive）

现实中会使用多个 candidate generation，来提供不同需求的候选（增加 diversity）：relevant, popular, trending

### Scoring

仍然使用 two tower nn，但是我们需要news的feature了

### reranking

- 可以使用独立的 ML 模型来分辨是否是 clickbait
- news 有没有 region restrict
- freshness 时效性
- misinformation
- duplicate/ near duplicate
- fair/ bias

## 主要的挑战

新用户：使用用户的 feature 来推荐 (age, gender, language, location)

新news：只有 metadata 和 news 本身，没有 interaction，我们可以给随机用户展示并收集 interaction

使用神经网络来便于 continuously fine-tuning