语言模型助手有三个目标：honest, helpful, harmless

找了很多人去标注哪些是有害的回答，哪些是无害的

helful 和 harmless 往往是对立的，因为什么都说不知道就能提高 harmless，同时也没有用了。从游戏 ELO 的情况来看，harmless 和 helpful 确实是对立的

做 alignment 会不会损害已经拥有的能力，实验显示对于比较大的模型来说，做了RLHF不会损失性能，相反会提升性能。alignment 本质上是让模型更讨人喜欢

在对话中，让人类去标注机器的发言的好坏，这些任务是尽可能的直观和熟悉的。更多是依赖人的直觉，但这种直觉是因人而异的，所以很难去评判标注人员的水平和结果好坏

数据收集主要有三个模型：
- context-distilled 52B language model
- Rejection sampling: 16-sample
- RLHF finetuned model: online iterative

对比两个模型哪个更好：win rate = $\frac{1}{1 + 10^{\frac{\Delta(Elo)}{400}}}$

$\Delta(Elo) \approx 174 \times \Delta(PM)$ Preference Modeling/奖励函数其实就是 10 换成了 e

# 做 preference model 需要三个步骤
1. pretrain language model
2. preference model pretraining (PMP): learning rate 0.1 on stackExchange, Reddit, Wikipedia
3. Finetuning on Human Feedback: lr 0.01 

scaling law: 数据指数级提升，模型 accuracy 性能线性提升

把 preference model 的 score 计算 $P(A > B) = \frac{1}{1 + e^{r_{PM}(B) - r_{PM}(A)}}$ 和真实的表现 accuracy 对比，发现模型预测的值是非常贴近 perfect calibration

当我们做第3步的时候，2里面的PM也需要及时更新来calibrate，这样才有利于训练的稳定性，如何stabilize呢？使用 Proximal Policy Optimization $r_{total} = r_{PM} - \lambda_{KL}D_{KL}(policy || policy_0)$ 尽可能提高 PM 的 reward，同时当前 policy 和上一个不能有太大的差异。作者发现后面这一项的数量级非常小，所以有可能根本不需要

RLHF 需要 PM 大小比较大，PM score 和 $\sqrt{D_{KL}(policy|policy_0)}$ 有比较好的线性关系，而且不同的模型参数也是保持平行关系，所以我们可以用很小的模型完整训练 + 几个大一点模型训练找到起点，来找到某个性能大概需要多少数据（模型更新的程度：$\sqrt{D_{KL}(policy|policy_0)}$）和模型大小。训练达到一定量级之后，能看到overfitting的现象，但是test PM仍然在涨

回答的质量很高的时候PM很难分辨两个回答的好坏。这个时候需要online重新收集一点数据，然后finetuning一下PM