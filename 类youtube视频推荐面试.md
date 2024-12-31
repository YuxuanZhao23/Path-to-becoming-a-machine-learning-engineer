# 前期提问，找到问题

1. 确认 business objective
2. 确认在什么情形下的推荐：主页/当前视频的相似视频
3. 确认用户和视频的 demographic
4. 收集什么数据？用户交互的数据
5. 有 playlist 功能吗？
6. 视频和用户的量级 10 billion
7. response 速度 < 200ms ？

# 翻译成 ML 问题

1. 提高 user click：小心 clickbait
2. 提高 completed video：小心短视频
3. 提高总观看时长：每天，每月，每年？
4. 最大化 related video 数量：怎么定义相关？
5. 输入：一个用户，所有视频；输出：一系列推荐的视频

# 模型

1. content-based filtering：用户喜欢看视频a，a和b类似，推荐b
   1. 好处：可以cold start，符合兴趣
   2. 坏处：不发掘新兴趣，怎么判断a和b类似？LLM之类的
2. collaborative filtering：用户ab类似，a喜欢视频x，给b推荐视频x
   1. 好处：不需要domain knowledge，发掘新兴趣，高效
   2. 坏处：cold start不了，niche兴趣处理不了（没有相似的用户）
3. 传统 matrix factorization model：做user embedding和video embedding相乘和实际观测做拟合
   1. loss 如果只计算观测值，那么没有惩罚错误
   2. loss 如果全都计算，本身把非观测值认为是用户不感兴趣的这就是错的
   3. weighted loss：解决方案
   4. optimize：SGD，WALS（固定一个，训练另一个，然后固定另一个♻️，更 parallel 和容易 converge）
   5. 优点：训练快，serving 快
   6. 缺点：只依赖interaction，不能处理新用户新内容
4. 双塔模型：一个用户塔一个视频塔，给两边输出的embedding计算inner product，通过Cross Entropy Loss 计算实际label 0/1的差距
   1. 可以高效 approximate top-k nearest neighbor
   2. 优点：利用上 user feature，可以处理新用户
   3. 缺点：serving 很慢，训练很慢

# 数据

1. 视频：id，长度，标题，tag，likes，views，language
2. 用户：id，名字，年龄，性别，城市，国家，语言，时区
3. interaction + context：user id，video id，type，timestamp，location
4. 处理：
   1. 对于 categorical：embedding/ one-hot/ bucketize 之后 one-hot
   2. 对于 数字：保持
   3. 文本（标题，搜索历史）：BERT
   4. 对于多个 tag：使用 CBOW，再全部 aggregate 在一起

# Evaluation

1. 离线指标：Precision@k（推荐的视频里面有多少比例是relevant）, mAP, Diversity
2. 在线指标：CTR, #completed, total watch time, explicit user feedback

# serving

1. candidate generation（recall）：双塔，只使用视频id（因为用户的embedding可以预先准备好），快速筛选，一般是多条通道
2. scoring：还是双塔，但是做的更大
3. reranking：最后调整 diversity, ad, clickbait, rule-based

# cold start

1. 新用户：demographics, random post
2. 新视频：heuristic (content/ YouTuber), random post to collect interaction