# PLM_task

# 1. 文本匹配（text_matching）
## 1.1 监督
### 1.1.1 PointWise（单塔）
1.正确开启训练后，终端会打印以下信息：
```
global step 10, epoch: 1, loss: 0.67510, speed: 4.25 step/s
global step 20, epoch: 1, loss: 0.65519, speed: 4.82 step/s
global step 30, epoch: 1, loss: 0.64859, speed: 4.81 step/s
global step 40, epoch: 1, loss: 0.61896, speed: 4.83 step/s
global step 50, epoch: 1, loss: 0.58966, speed: 4.76 step/s
global step 60, epoch: 1, loss: 0.55598, speed: 4.73 step/s
global step 70, epoch: 1, loss: 0.53062, speed: 4.74 step/s
global step 80, epoch: 1, loss: 0.50343, speed: 4.71 step/s
global step 90, epoch: 2, loss: 0.47102, speed: 4.93 step/s
global step 100, epoch: 2, loss: 0.44601, speed: 4.80 step/s
...
```
在 `logs/comment_classify` 文件下将会保存训练曲线图：

![11](https://github.com/Tracyyytao/PLM_task/blob/main/text_matching/supervised/assets/Model%20Performance%20(1).png?raw=true)

2.完成模型训练后，运行 `inference_pointwise.py` 以加载训练好的模型并应用，得到以下推理结果：
```
tensor([[ 1.9092, -2.6020]], device='cuda:0')
```
### 1.1.2  DSSM（双塔）
1.正确开启训练后，终端会打印以下信息：
```
global step 10, epoch: 1, loss: 0.74789, speed: 4.67 step/s
global step 20, epoch: 1, loss: 0.67472, speed: 4.53 step/s
global step 30, epoch: 1, loss: 0.56567, speed: 4.81 step/s
global step 40, epoch: 1, loss: 0.50395, speed: 4.90 step/s
...
```
在 `logs/comment_classify` 文件下将会保存训练曲线图：

![22](https://github.com/Tracyyytao/PLM_task/blob/main/text_matching/supervised/assets/Model%20Performance%20(2).png?raw=true)

2.运行 `inference_dssm.py`，得到下面结果：
```
Used 0.3007240295410156s.
[
    ('衣服', 0.7229569554328918),
    ('酒店', 0.7043281197547913),
    ('蒙牛', 0.6797731518745422),
    ('洗浴', 0.6788960695266724),
    ('电器', 0.6734145879745483),
    ('水果', 0.6660837531089783),
    ('电脑', 0.6475979089736938),
    ('手机', 0.635999321937561),
    ('平板', 0.5878052711486816),
    ('书籍', 0.5437681674957275)
]
```
### 1.1.3  Sentence Transformer（双塔）
1.正确开启训练后，终端会打印以下信息：
```
Step 10, Epoch 1, Loss: 0.7026, Speed: 2.44 steps/s
Step 20, Epoch 1, Loss: 0.6451, Speed: 2.62 steps/s
Step 30, Epoch 1, Loss: 0.6210, Speed: 2.60 steps/s
Step 40, Epoch 1, Loss: 0.5955, Speed: 2.58 steps/s
Step 50, Epoch 1, Loss: 0.5885, Speed: 2.61 steps/s
Step 60, Epoch 1, Loss: 0.5940, Speed: 2.61 steps/s
Step 70, Epoch 1, Loss: 0.5893, Speed: 2.59 steps/s
Step 80, Epoch 1, Loss: 0.5712, Speed: 2.61 steps/s
Step 90, Epoch 2, Loss: 0.4719, Speed: 2.72 steps/s
...
```
在 `logs/comment_classify` 文件下将会保存训练曲线图：

![33](https://github.com/Tracyyytao/PLM_task/blob/main/text_matching/supervised/assets/Model%20Performance%20(3).png?raw=true)

2.运行 `inference_sentence_transformer.py`，函数会输出所有类别里「匹配通过」的类别及其匹配值，得到下面结果：
```
Used 0.22788548469543457s.
[
    ('平板', 1.1538971234932689), 
    ('电脑', 0.8854678934975480)
]
```
## 1.2 无监督
1.正确开启训练后，终端会打印以下信息：
```
...
global step 140, epoch: 1, loss: 0.12339, speed: 2.52 step/s
global step 150, epoch: 1, loss: 0.11530, speed: 2.52 step/s
global step 160, epoch: 1, loss: 0.10818, speed: 2.52 step/s
global step 170, epoch: 1, loss: 0.10186, speed: 2.53 step/s
global step 180, epoch: 1, loss: 0.09625, speed: 2.52 step/s
global step 190, epoch: 1, loss: 0.09134, speed: 2.54 step/s
global step 200, epoch: 1, loss: 0.08684, speed: 2.52 step/s
Evaluation precision: 0.54226, recall: 0.98819, F1: 0.70026, spearman_corr: 
0.47653
best F1 performance updated: 0.00000 --> 0.70026
global step 210, epoch: 1, loss: 0.11488, speed: 2.36 step/s
global step 220, epoch: 1, loss: 0.11552, speed: 2.35 step/s
...
```
在 `logs/LCQMC` 文件下将会保存训练曲线图：

![44](https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/assets/ERNIE-ESimCSE.png?raw=true)

2.完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，得到以下推理结果：
`[0.2931939363479614, 0.9064270257949829]`

# 2. 信息抽取（UIE）
1.模型训练
正确开启训练后，终端会打印以下信息：
```
global step 10, epoch: 1, loss: 0.00229, speed: 3.79 step/s
global step 20, epoch: 1, loss: 0.00173, speed: 4.31 step/s
global step 30, epoch: 1, loss: 0.00138, speed: 4.33 step/s
global step 40, epoch: 1, loss: 0.00121, speed: 4.33 step/s
global step 50, epoch: 1, loss: 0.00109, speed: 4.31 step/s
global step 60, epoch: 1, loss: 0.00099, speed: 4.32 step/s
global step 70, epoch: 2, loss: 0.00092, speed: 4.60 step/s
global step 80, epoch: 2, loss: 0.00084, speed: 4.34 step/s
global step 90, epoch: 2, loss: 0.00079, speed: 4.31 step/s
...
```
在 `logs/UIE Base.png` 文件中将会保存训练曲线图：

![Model Performance](https://github.com/Tracyyytao/PLM_task/blob/main/UIE/assets/Model%20Performance.png?raw=true)

2.模型预测
完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，能得到以下推理结果：
```
[+] NER Results: 
{'人物': ['谭孝曾', '谭元寿']}
[+] Information-Extraction Results: 
{'谭孝曾': {'父亲': ['谭元寿']}, '谭元寿': {'父亲': []}}
```

# 3.  Prompt任务（Prompt_task）
## 3.1 PET（基于人工定义 prompt pattern 的方法）
1.模型训练
正确开启训练后，终端会打印以下信息：
```
Prompt is -> 这是一条{MASK}评论：{textA}。
Map: 100%|█████████████████████████████| 63/63 [00:00<00:00, 2088.88 examples/s]
Map: 100%|███████████████████████████| 590/590 [00:00<00:00, 2602.00 examples/s]
global step 5, epoch: 0, loss: 3.11866, speed: 7.65 step/s
global step 10, epoch: 1, loss: 2.69449, speed: 16.02 step/s
global step 15, epoch: 1, loss: 2.16243, speed: 15.25 step/s
global step 20, epoch: 2, loss: 1.81705, speed: 15.99 step/s
global step 25, epoch: 3, loss: 1.55081, speed: 16.04 step/s
global step 30, epoch: 3, loss: 1.32731, speed: 15.72 step/s
global step 35, epoch: 4, loss: 1.15957, speed: 15.88 step/s
global step 40, epoch: 4, loss: 1.01925, speed: 16.13 step/s
Evaluation precision: 0.77000, recall: 0.71000, F1: 0.70000
best F1 performence has been updated: 0.00000 --> 0.70000
Each Class Metrics are: {'书籍': {'precision': 0.97, 'recall': 0.73, 'f1': 
0.83}, '平板': {'precision': 0.62, 'recall': 0.6, 'f1': 0.61}, '手机': 
{'precision': 0.0, 'recall': 0.0, 'f1': 0}, '水果': {'precision': 0.98, 
'recall': 0.73, 'f1': 0.84}, '洗浴': {'precision': 0.79, 'recall': 0.69, 'f1': 
0.74}, '电器': {'precision': 0, 'recall': 0.0, 'f1': 0}, '电脑': {'precision': 
0.91, 'recall': 0.31, 'f1': 0.47}, '蒙牛': {'precision': 1.0, 'recall': 0.05, 
'f1': 0.1}, '衣服': {'precision': 0.46, 'recall': 1.0, 'f1': 0.63}, '酒店': 
{'precision': 1.0, 'recall': 0.91, 'f1': 0.95}}
...
```
在 `logs/sentiment_classification` 文件中将会保存训练曲线图：

![BERT](https://github.com/Tracyyytao/PLM_task/blob/main/prompt_tasks/PET/assets/BERT.png?raw=true)

2.模型预测
完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，能得到以下推理结果：
```
Prompt is -> 这是一条{MASK}评论：{textA}。
Used 0.22090387344360352s.
inference label(s):
['酒店', '洗浴']
```
## 3.2 p-tuning（机器自动学习 prompt pattern 的方法）
1.模型训练
正确开启训练后，终端会打印以下信息：
```
global step 5, epoch: 1, loss: 3.08010, speed: 7.69 step/s
global step 10, epoch: 2, loss: 2.05724, speed: 15.12 step/s
global step 15, epoch: 3, loss: 1.43877, speed: 16.02 step/s
global step 20, epoch: 4, loss: 1.08729, speed: 16.33 step/s
Evaluation precision: 0.76000, recall: 0.72000, F1: 0.70000
best F1 performence has been updated: 0.00000 --> 0.70000
Each Class Metrics are: {'书籍': {'precision': 0.97, 'recall': 0.82, 'f1': 
0.89}, '平板': {'precision': 0.55, 'recall': 0.79, 'f1': 0.65}, '手机': 
{'precision': 0.33, 'recall': 0.06, 'f1': 0.1}, '水果': {'precision': 0.97, 
'recall': 0.63, 'f1': 0.76}, '洗浴': {'precision': 0.71, 'recall': 0.69, 'f1': 
0.7}, '电器': {'precision': 0, 'recall': 0.0, 'f1': 0}, '电脑': {'precision': 
0.5, 'recall': 0.12, 'f1': 0.2}, '蒙牛': {'precision': 1.0, 'recall': 0.21, 
'f1': 0.35}, '衣服': {'precision': 0.58, 'recall': 0.99, 'f1': 0.73}, '酒店': 
{'precision': 1.0, 'recall': 0.89, 'f1': 0.94}}
```
在 `logs/sentiment_classification` 文件下将会保存训练曲线图：

![BERT](https://github.com/Tracyyytao/PLM_task/blob/main/prompt_tasks/p-tuning/assets/BERT.png?raw=true)

2.模型预测
完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，能得到以下推理结果：
```
Used 0.32077670097351074s.
苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的 -> 洗浴
这破笔记本速度太慢了，卡的不要不要的 -> 平板
```

# 4. 文本分类（text_classification）
1.模型训练
正确开启训练后，终端会打印以下信息：
```
global step 10, epoch: 1, loss: 2.01464, speed: 11.28 step/s
global step 20, epoch: 1, loss: 1.86602, speed: 16.46 step/s
global step 30, epoch: 2, loss: 1.69751, speed: 17.34 step/s
global step 40, epoch: 2, loss: 1.49696, speed: 16.94 step/s
global step 50, epoch: 2, loss: 1.30735, speed: 19.60 step/s
Evaluation precision: 0.85000, recall: 0.84000, F1: 0.84000
best F1 performence has been updated: 0.00000 --> 0.84000
Each Class Metrics are: {0: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}, 1: 
{'precision': 0.7, 'recall': 1.0, 'f1': 0.82}, 2: {'precision': 1.0, 'recall': 
0.83, 'f1': 0.91}, 3: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}, 4: 
{'precision': 0.77, 'recall': 0.62, 'f1': 0.69}, 5: {'precision': 1.0, 'recall':
1.0, 'f1': 1.0}, 6: {'precision': 0, 'recall': 0.0, 'f1': 0}, 7: {'precision': 
0.56, 'recall': 0.83, 'f1': 0.67}}
...
```
在 `logs/comment_classify` 文件中将会保存训练曲线图：

![BERT](https://github.com/Tracyyytao/PLM_task/blob/main/text_classification/assets/BERT.png?raw=true)

2.模型预测
完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，能得到以下推理结果：
```
res: 
[4, 4, 5]
```

# 5. 强化学习&语言模型（RLHF）
## 5.1  基于中文情感识别模型的正向评论生成机器人（No Human Reward）
1.正常启动训练后，终端会打印如下数据:
```
Random Sample 5 text(s) of model output:
1. 这部电影很浅 不 精 实 。 一 本 书 买 到 后 很 失 望 [SEP] [SEP]
2. 这部电影很幻 化 成 旷 世 奇 谭 杀 青 ， 两 年 后 全 剧 终
3. 这次购物总的来说体验很正 。 感 触 很 深 也 很 有 功 底 。 书 还 是 很
4. 说实话，真的很般 地 一 般 。 而 且 越 来 越 贵 ， 以 前 百 叶
5. 刚收到货，感觉音 效 不 错 ， 就 是 电 池 续 航 不 咋 滴 。 [SEP]
  1%|▎                                        | 1/157 [00:45<1:58:46, 45.68s/it]epoch 1 mean-reward: 0.660504937171936
Random Sample 5 text(s) of model output:
1. 说实话，真的很般 般 ， 咱 就 是 吃 顿 饭 来 接 受 不 了 。 这
2. 这次购物总的来说体验很可 口 [SEP] ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
3. 说实话，真的很般 ， 还 加 了 20 五 ， 猪 脚 炖 土 豆 不 脆 ，
4. 刚收到货，感觉~ ~ ~ ~ ~ 因 为 不 常 用 什 么 电 脑 ， 所
5. 这次购物总的来说体验很是 翻 开 的 书 有 点 厚 也 少 了 点 这 种 书 的
  1%|▌                                        | 2/157 [01:31<1:58:49, 45.99s/it]epoch 2 mean-reward: 0.6786868572235107
```
在 `logs/PPO-Sentiment-Zh.png` 下会保存模型训练过程中的各个指标变化（包括 reward 变化曲线）：

![PPO](https://github.com/Tracyyytao/PLM_task/blob/main/PLHF/assets/PPO-Sentiment-Zh.png?raw=true)

## 5.2  基于人工打分的评论生成机器人（With Human Reward）
启动标注平台`terminal_main.py`，可以在终端看到模型的生成结果，通过人工输入 reward 以迭代模型：

![terminal](https://github.com/Tracyyytao/PLM_task/blob/main/PLHF/assets/terminal.png?raw=true)

## 5.3  基于人工排序训练 Reward Model
成功开始训练后，终端会打印以下信息：
```
...
global step 100, epoch: 1, loss: 0.32152, speed: 1.49 step/s
global step 110, epoch: 1, loss: 0.31962, speed: 1.46 step/s
global step 120, epoch: 1, loss: 0.31617, speed: 1.46 step/s
global step 130, epoch: 1, loss: 0.31261, speed: 1.40 step/s
global step 140, epoch: 1, loss: 0.30733, speed: 1.40 step/s
global step 150, epoch: 1, loss: 0.30515, speed: 1.43 step/s
global step 160, epoch: 1, loss: 0.30182, speed: 1.44 step/s
global step 170, epoch: 1, loss: 0.30007, speed: 1.43 step/s
global step 180, epoch: 1, loss: 0.29843, speed: 1.43 step/s
global step 190, epoch: 1, loss: 0.29636, speed: 1.45 step/s
global step 200, epoch: 1, loss: 0.29429, speed: 1.47 step/s
Evaluation acc: 0.54348
best F1 performence has been updated: 0.00000 --> 0.54348
...
```
在 `logs/Model Performance.png` 会存放训练曲线图：

![Model Performance](https://github.com/Tracyyytao/PLM_task/blob/main/PLHF/assets/Model%20Performance.png?raw=true)

运行预测脚本，可以看到训练后的模型的打分效果：
```
tensor([[ 3.2337],
        [-6.1985]], grad_fn=<AddmmBackward0>)
```

# 6. 文本生成（answer_generation)
## 6.1 中文问答模型（T5-Small）
1.正确开启训练后，终端会打印以下信息：
```
Map: 100%|███████████████████████| 14520/14520 [00:09<00:00, 1526.76 examples/s]
Map: 100%|███████████████████████████| 984/984 [00:00<00:00, 1525.13 examples/s]
global step 10, epoch: 1, loss: 9.39759, speed: 4.67 step/s
global step 20, epoch: 1, loss: 9.39064, speed: 5.52 step/s
global step 30, epoch: 1, loss: 9.38868, speed: 5.43 step/s
global step 40, epoch: 1, loss: 9.38350, speed: 5.48 step/s
global step 50, epoch: 1, loss: 9.37548, speed: 5.47 step/s
global step 60, epoch: 1, loss: 9.36682, speed: 5.51 step/s
global step 70, epoch: 1, loss: 9.35632, speed: 5.53 step/s
global step 80, epoch: 1, loss: 9.34300, speed: 5.52 step/s
global step 90, epoch: 1, loss: 9.32656, speed: 5.53 step/s
global step 100, epoch: 1, loss: 9.30552, speed: 5.45 step/s
global step 110, epoch: 1, loss: 9.27951, speed: 5.48 step/s
...
```
在 `logs/DuReaderQG` 文件下将会保存训练曲线图：

![11](https://github.com/Tracyyytao/PLM_task/blob/main/answer_generation/assets/Model%20Performance.png?raw=true)

2.完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：
```
Q: "治疗宫颈糜烂的最佳时间"
C: 
"专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，
如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，
术前3天禁同房，有生殖道急性炎症者应治好后才可进行。"
A: "答案：3-7日"
```

## 6.2 Filling 模型（T5-Small）
1.正确开启训练后，终端会打印以下信息：
```
Map: 100%|█████████████████████| 350134/350134 [01:57<00:00, 2969.48 examples/s]
Map: 100%|███████████████████████| 38904/38904 [00:13<00:00, 2967.30 examples/s]
global step 10, epoch: 1, loss: 9.51035, speed: 4.68 step/s
global step 20, epoch: 1, loss: 9.51205, speed: 5.60 step/s
global step 30, epoch: 1, loss: 9.51319, speed: 5.49 step/s
global step 40, epoch: 1, loss: 9.51603, speed: 5.48 step/s
global step 50, epoch: 1, loss: 9.51272, speed: 5.50 step/s
global step 60, epoch: 1, loss: 9.51306, speed: 5.50 step/s
global step 70, epoch: 1, loss: 9.51283, speed: 5.50 step/s
global step 80, epoch: 1, loss: 9.51172, speed: 5.54 step/s
global step 90, epoch: 1, loss: 9.51086, speed: 5.46 step/s
...
```
训练曲线图如下：

![22](https://github.com/Tracyyytao/PLM_task/blob/main/data_augment/filling_model/assets/T5-Base-Chinese.png?raw=true)

2.成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，得到以下推理结果：
```
maksed text: 
['"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]
位置的文本是：']
output: ['extra0的的人，的一部的的的']
```

# 7. 大模型应用
1.文本分类
运行`llm_classification.py`文件，结果如下：
```
>>> sentence：
加拿大（英语/法语：Canada），首都渥太华，位于北美洲北部。东临大西洋，
西濒太平洋，西北部邻美国阿拉斯加州，南接美国本土，北靠北冰洋。
气候大部分为亚寒带针叶林气候和湿润大陆性气候，北部极地区域为极地长寒气候。
>>> inference answer：
国家

>>> sentence：
《琅琊榜》是由山东影视传媒集团、山东影视制作有限公司、北京儒意欣欣影业投资有限公司、
北京和颂天地影视文化有限公司、北京圣基影业有限公司、东阳正午阳光影视有限公司联合出品，
由孔笙、李雪执导，胡歌、刘涛、王凯、黄维德、陈龙、吴磊、高鑫等主演的古装剧。
>>> inference answer：
电视剧
...
```
2.文本匹配
运行`llm_text_matching.py`文件，结果如下：
```
>>> sentence: ('如何修改头像', '可以通过上传图片修改头像吗')
>>> inference answer: 是

>>> sentence: ('王者荣耀司马懿连招', '王者荣耀司马懿有什么技巧')
>>> inference answer: 是

>>> sentence: ('王者荣耀司马懿连招', '历史上司马懿真的被诸葛亮空城计骗了吗')
>>> inference answer: 不是
```
3.信息抽取
运行`llm_information_extraction.py`文件，结果如下：
```
>>> sentence：张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，
中国内地男演员。1997年至2006年服役于北京军区政治部战友话剧团。
2006年，主演军事励志题材电视剧《士兵突击》。
>>> inference answer:{'姓名':['张译']，'性别':['男']，
'出生日期':['1978年2月17日']，'出生地点':['黑龙江省哈尔滨市'，
'职业':['男演员']，'获得奖项':['原文中未提及']}
...
```
