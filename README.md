# PLM_task

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
