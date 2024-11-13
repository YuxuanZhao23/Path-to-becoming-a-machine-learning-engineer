# 自我介绍

Hi, I'm Yuxuan, a senior software engineer at Fidelity. I'm passionate about machine learning and have a strong track record of Computer Vision, Natural Language Processing, Human-Computer Interaction. I'm excited about the opportunity to contribute my skills to Evident.

# 为什么离职

I enjoy working at Fidelity, but I am looking forward to more challenge and bigger scope/ impact. So I think joining a startup is a good idea

# 最大的挑战

- S：Fidelity's Trading Dashboard will feed news
- T：the news retreival quality is not good
- A：look into the training process, We selected news that had impression but no user interaction as negative samples in the recall phase
- R：Half of the negative samples are negative samples within a batch, and the other half are samples that did not pass the sorting stage. improve retrieval

inverted index

在前端有埋点，一旦曝光就会立刻用实时流进行处理（Kafka 队列 + Flink 实时计算哈希值），然后尽快写进 bloom filter，因为两次刷新的间隔可能只有几分钟，所以bloom filter的更新一定要及时