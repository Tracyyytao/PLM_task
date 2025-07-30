# PLM_task

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
