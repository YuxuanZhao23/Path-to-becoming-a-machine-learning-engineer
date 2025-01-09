let's think step by step 能让语言模型显著提升数学能力

Manual COT: 手动构造 few-shot 的问题和答案

Auto COT: 用多样性采样问题，然后加上 let's think step by step，让推理模型生成推理步骤和答案，然后把这些拼接后当成 few shot 再传给语言模型。Auto COT 节省了构造问题答案的时间成本，同时效果比 Manual COT 还要好

- system-1 task 就是 LM 很容易做好的任务，人也是能很快和很直观做到的
- 与之相对的 system-2 task 就是需要很慢的思考，往往需要很多步骤的问题
- COT 的定义就是一些中间步骤的短句子
- 因为需要语言模型输出更多的中间步骤，所以语言模型需要输出更多的token，间接增大了计算量，更有可能得到正确结果
- COT 的好处是有更多的可解释性