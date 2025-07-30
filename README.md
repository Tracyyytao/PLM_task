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
在 `logs/UIE Base.png` 文件中将会保存训练曲线图：

![BERT](https://github.com/Tracyyytao/PLM_task/blob/main/text_classification/assets/BERT.png?raw=true)

2.模型预测
完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用，能得到以下推理结果：
```
res: 
[4, 4, 5]
```
