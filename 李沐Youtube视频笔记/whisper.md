数据预处理：把音频 resample 成 16000 Hz，以及 80 channel log Mel spectrogram，每个样本都是被切成 30s。所以一个输入就是 3000 x 80。通过一个 CNN kernel 25 ms, stride 10 ms（为什么选定 25ms，是因为这个是说一个词大概需要的时间）变成 80 x 1500。然后接一个标准的 seq2seq 的 encoder-decoder 学习。

作者想要用一个单一的模型做多个语音的任务：transcription, translation, voice activity detection, alignment, language identification。模型本身是标准的语言模型概念，也就是预测下一个词。用任务的控制流（if-else 业务逻辑）来判断需要如何处理这些输入

训练细节：数据格式 FP16，AdamW, gradient norm clipping, decay learning rate, 2-3 epoch, 因为扫的次数有限，所以 overfit 不是问题

Word Error Rate = $\frac{\text{substitution} + \text{deletion} + \text{insertion}}{\text{substitution} + \text{deletion} + \text{Corrected}} = \frac{\text{substitution} + \text{deletion} + \text{insertion}}{N}$

作者在刷榜的时候给最终输出加上了一个 text normalizer 来规范输出和数据集结果的格式，减少 WER。实际使用中/训练中是不需要这个 text normalizer

如何处理较长的音频？训练只用了30s，如果处理一个小时的音频，错误会累加。使用 5 beam search 和根据 log probability 和 gzip 规则来放宽 temperature 来增加一点生成的多样性，来避免 repetition looping