import random
import json
import csv

datasets=[]
#qa
# with open(f"datasets/BOSS/QuestionAnswering/searchqa/test-all.json",'r') as f:
#     lines = f.readlines()
#     for line in lines:#别的数据集也是按行读吗
#         json_data = json.loads(line)
#         datasets.append(json_data)

#sa
# with open("datasets/BOSS/SentimentAnalysis/sst5/test-all.tsv", 'r', encoding='utf-8') as input_file:
#     reader = csv.reader(input_file, delimiter='\t')
#     datasets = list(reader)

with open("datasets/BOSS/ToxicDetection/toxigen/test-all.tsv", 'r', encoding='utf-8') as input_file:
    reader = csv.reader(input_file, delimiter='\t')
    datasets = list(reader)


random.seed(42)
random.shuffle(datasets)
train_data = datasets[:300]
test_data = datasets[300:]

# qa
# with open('datasets/BOSS/QuestionAnswering/searchqa/train.json', 'w', encoding='utf-8') as file:
#     for data in train_data:
#         json.dump(data, file, separators=(',', ':')) 
#         file.write('\n')
# with open('datasets/BOSS/QuestionAnswering/searchqa/test.json', 'w', encoding='utf-8') as file:
#     for data in test_data:
#         json.dump(data, file, separators=(',', ':')) 
#         file.write('\n')

# sa
with open("datasets/BOSS/ToxicDetection/toxigen/test.tsv", 'w', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(test_data)

with open("datasets/BOSS/ToxicDetection/toxigen/train.tsv", 'w', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerows(train_data)