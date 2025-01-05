# 前期提问，找到问题

1. 确认想要最大化 revenue？
2. 假设 ad placed on timeline，每次点击都带来相同的 revenue
3. 同一个广告可以有多次 impression 吗？可以，但是一般有 fatigue period，也就是说如果用户一直没有点击，在一段时间内不推送这个广告
4. 有没有 hide 或者 block ad 的功能
5. 训练数据：用户和广告的交互
6. 怎么定义 negative example？展示了一会儿但是没有被点击/没有点击的impression/hide ad
7. 有没有 continual learning？因为5分钟的延迟都会影响性能

# 翻译成 ML 问题

用 pointwise Learning to Rank (LTR) 预测一个广告被点击的概率

# 数据

1. 广告：id, advertiser id, group id, campaign id, category, subcategory, image/ video
2. user: id, user name, age, gender, city, country, language, time zone
3. interaction: user id, ad id, interaction type, dwell time, location, timestamp