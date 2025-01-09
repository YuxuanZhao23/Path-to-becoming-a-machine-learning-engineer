# Generative Pre-Training GPT

怎么在没有标号的数据集上训练？
- 文章里说自己用到的算法是 semi-supervised，后来大多数人使用 self-supervised 这个词来指示相同的含义
- 给定前 k 个词（并不是前面所有的词！k是一个常数）预测下一个词的出现的概率的 log
- 模型使用的是 transformer 的 decoder，在当前预测词以及后面的内容会做一个 mask
- 与 BERT 的主要区别在于 BERT 是做完形填空，但是 GPT 是看不到后面的内容的

如何在下游任务微调？
- 可以用 softmax 计算一个概率，然后计算log的和
- 同时用一个超参数乘上之前的 loss 效果会更好

怎么把下游任务表示成一段文字和一个标号？
- classification: [start] text [extract] => Transformer => 新的 Linear
- Entailment: [start] premise [delim] hypothesis [extract] => Transformer => Linear 三分类
- similarity: 
$$
\left.
\begin{array}{c}
\text{[start] text1 [delim] text2 [extract] => Transformer}\\
\text{[start] text2 [delim] text1 [extract] => Transformer}
\end{array}
\right \}
+ \rightarrow \text{Linear}
$$
- Multiple Choice:
$$
\left.
\begin{array}{c}
\text{[start] context [delim] answer1 [extract] => Transformer => Linear}\\
\text{[start] context [delim] answer2 [extract] => Transformer => Linear}\\
\text{[start] context [delim] answer3 [extract] => Transformer => Linear}
\end{array}
\right \}
$$

作者发现 transformer 能够有更 robust 的 transfer performance，尤其是对于比较长的数据来说。GPT1模型大小是12层decoder，每一层768，12个注意力头（大概有1亿可学习的参数）

# GPT 2

但是GPT1打不过BERT，单纯加数据和模型容量也和BERT差不多，所以作者把卖点集中在 zero shot 上。也就是说训练好的模型可以直接做下游任务而不需要收集下游任务的有标号的数据（省时间省钱也不需要训练/微调）

因为要 zero-shot，所以下游任务的时候不能再引入模型之前没看过的符号。所以下游任务就构造成自然语言：比如说translate to french, answer the question。这个神奇的句子就被叫做 prompt

除了使用书籍和 Wikipedia 以外，作者尝试找到更多的数据。比如 common crawl 这种公开的爬虫爬取数据集，但是作者说里面的内容 mostly unintelligible。所以作者使用的是 reddit 里面有至少 3 个 karma 的帖子

GPT2 做的最大模型有 48 层，每一层 1600，有15亿个可学习的参数。在阅读理解这个任务上还可以，翻译和总结差得比较多，QA上面差非常非常多

# GPT 3

人类其实在做新任务的时候也一般是 few shot 的，只不过语言模型的样本有效性特别低，需要大量的样本来学习。所以放弃了 GPT 2 推的 zero shot，还是关注在模型的有效性（要刷榜）

GPT 3 有 1750 亿的参数，所以即使是微调也不会做 gradient update（也做不了）所以例子是以 prompt 的形式给的

一般来说我们认为模型能力太强的话容易对于数据过拟合。但是作者发现在超大数据和超大模型中反而不会过拟合，可以大胆使用较大的batch size，这样机器的计算并行度会好一点（减少一点通讯的损耗比例）

本文在越大的模型上用的是越小的学习率，比较反直觉。

common crawl 数据清理：
1. 用一个二分类模型，把之前 reddit 的数据当作正样本，common crawl 当作负样本来学习，然后filter出高质量的内容
2. Locality-sensitive hashing 算法 deduplication