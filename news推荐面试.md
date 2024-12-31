# 前期提问，找到问题

1. 确认 business objective
2. 确认在什么情形下的推荐：展示unseen posts 和 comments
3. 内容的形式：文本，图片，视频，结合？
4. most engaging content 放在最前面？
5. 有什么交互方式？交互的优先级定义(click, like, share, comment, hide, block)
6. 用户数量，反应速度

# 翻译成 ML 问题

1. 增加 dwell time/ click：implicit signal 更多，但是不一定反映真实喜好(clickbait)
2. 增加 reaction：数据少，但反映真实喜好
3. weighted both
4. 输入：用户，新闻；输出：一系列的推荐新闻

# 模型

pointwise learn to rank (LTR) 给不同的交互不同的value，模型就可以预测每种交互出现的概率乘上value得到一个总的score

1. 可以给每个任务(click, like, share, comment)做一个 deep neural network：这样很贵，reaction数量很少的类型训练不了
2. multi-task DNN：一个模型输出多个reaction的出现概率。那怎么选择loss function呢？可以将所有的loss相加（dwell的需要乘一个超参数平衡一下）
3. 如何处理 passive users 很多？训练 skip 和 dwell-time

# 数据

1. post：id，文本内容，tag，mention，image，video，timestamp
2. 用户：id，名字，年龄，性别，城市，国家，语言，时区
3. interaction + context: user id, post id, type, value, location, timestamp
4. friendship/ connection: user id 1/2, time, close?, family?
5. 处理：
   1. 文本：bert
   2. 图片/视频：clip, resnet
   3. hashtag: 不需要 transformer，因为不要上下文，只需要 TF-IDF 或者 word2vec 就好
   4. post age：bucketize + one-hot
   5. affinity: 友谊长度，关系，reaction rate

# Evaluation

1. 离线指标：precision, recall, ROC curve 来理解 true positive 和 false positive rate
2. 在线指标：CTR, reaction rate, total time, user survey's satisfaction

# Serving

1. retrieval：取回没看过的 post
2. ranking：打分
3. re-ranking：额外的business logic 或者 interest filtering