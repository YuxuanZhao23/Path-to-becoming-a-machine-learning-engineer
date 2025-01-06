inverted index 
- 是用在搜索的常规提速方法
- 每一个关键词都会有一个列表记录所有出现的位置
- 缺点：只能精准匹配

semantic matching
- 用双塔模型把 query 和 document 变成 embedding
- 用 pairwise cosine similarity 训练
- inference 的时候找 approximate nearest neighbor，因为简单做内积成本也太高了
- 因为想做 ANN 所以一般必须在 Euclidean space 里面，只能用 cos similarity，复杂 QA 可能不够用了。而且有的时候我们就是需要 exact match 比如用户搜索 iPhone 14，就最好不要返回 iPhone 13 的信息

本文非常类似于 Differentiable Search Index DSI，核心观点是模型不需要和document database，把 document 的内容和 id 直接学习到模型里面，然后给定一个输入就可以直接返回id。这样做的结果是对于很小的 document database 都需要很长的 recall 时间，实践中很大的模型不可能这么做。另外如果有新增加的 document 也很难处理，一方面可能会影响到大量的 id，另一方面模型重新训练成本太高（但不训练如何把新的内容放到模型里面呢？）文章另一处受到诟病的地方是没有使用常规的retrieval dataset，而是使用Wikipedia QA，比较小而且任务不对。QA往往query比搜索要长很多

使用的是 k-means 的 hierarchical id，这样的好处是标号是有内涵的，每一个数字代表属于哪一个子类

query generation: 使用 DocT5Query 来做 random sampling，从 doc 到 query 是这么做的，没有用beam search是因为找最优的多样性就比较差