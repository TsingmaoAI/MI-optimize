---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
- multiple-choice
- question-answering
language:
- zh
pretty_name: C-Eval
size_categories:
- 10K<n<100K
---

C-Eval is a comprehensive Chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels. Please visit our [website](https://cevalbenchmark.com/) and [GitHub](https://github.com/SJTU-LIT/ceval/tree/main) or check our [paper](https://arxiv.org/abs/2305.08322) for more details.

Each subject consists of three splits: dev, val, and test.  The dev set per subject consists of five exemplars with explanations for few-shot evaluation. The val set is intended to be used for hyperparameter tuning. And the test set is for model evaluation. Labels on the test split are not released, users are required to submit their results to automatically obtain test accuracy. [How to submit?](https://github.com/SJTU-LIT/ceval/tree/main#how-to-submit)

### Load the data
```python
from datasets import load_dataset
dataset=load_dataset(r"ceval/ceval-exam",name="computer_network")

print(dataset['val'][0])
# {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}
```
More details on loading and using the data are at our [github page](https://github.com/SJTU-LIT/ceval#data).

Please cite our paper if you use our dataset.
```
@article{huang2023ceval,
title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models}, 
author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
journal={arXiv preprint arXiv:2305.08322},
year={2023}
}
```


