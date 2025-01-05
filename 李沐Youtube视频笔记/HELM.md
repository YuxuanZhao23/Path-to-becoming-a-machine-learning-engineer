# 评价语言模型的 matrix

- Exact Match
- Quasi-Exact Match 改变大小写之后可以 exact match
- F1 score
  - Precision = $\frac{TP}{TP + FP}$ 这种错误是划多了一些正确的（要求太宽）
  - Recall = $\frac{TP}{TP + FN}$ 这种错误是漏了一些正确的（要求太窄）
  - F1 = $\frac{2 \times P \times R}{P + R}$
- RR@K: 模型排序，如果正确结果的rank $\leq$ K 的时候返回 $\frac{1}{rank}$，否则返回 0
- NDCG@K Normalized Discounted Cumulative Gain
  - DCG@K = $\displaystyle\sum_{i=1}^K \frac{grade(d_i)}{log_2(i+1)}$ 文档的排名越前面，分母就越小
  - normalize 的意思就是算出一个DCG@K最优解作为upper bound，然后作为分母
- ROUGE-2：$\frac{\text{overlapped bigram}}{\text{total bigram}}$
- BPB 文本的压缩能力 byte tokenize