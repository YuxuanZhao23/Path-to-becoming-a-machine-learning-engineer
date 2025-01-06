- Supervised Fine-Tuning SFT: 在 prompt database 里面随机 sample 一个，然后人类写出模型预期返回的内容。拼成一段话来训练模型
- Reinforcement Learning Human Feedback RLHF: 一个 prompt beam search 多个不同的 output，让人去排序回答的好坏
- 模型上线之后，会收集用户的 prompt，然后根据 user id 来放进 train, validate, test 里面，因为同一个用户可能问的方式是类似的

# RLHF

把最后一层换成一个线性层，直接输出一个数字。而计算这个数字的 loss 使用的是 ranking 里面常见的 pairwise ranking loss。K 用了 9 而不是 4，因为这样单位时间人可以标注的数据更多了。loss 套了 sigmoid 和 log

训练了一个 $r_\theta$ 来学习人的排序，这样就不用一直找人来标新的模型输出了。强化模型用的是 PPO-ptx，用 KL divergence 来 regularize 避免模型改变得太多，这样模型返回的 y 不会变化太大，那么 $r_\theta$ 就能估计得比较准。然后还是加上了语言模型的训练损失来避免过拟合任务而丢失掉模型原本的能力（Catastrophic interference）