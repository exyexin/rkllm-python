import pandas as pd

# 读取 Parquet 文件
train_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/train-00000-of-00001.parquet')
validation_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/validation-00000-of-00001.parquet')
test_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/test-00000-of-00001.parquet')

# 将 DataFrame 转换为 JSON 格式
train_json = train_df.to_json(orient='records', lines=True)
validation_json = validation_df.to_json(orient='records', lines=True)
test_json = test_df.to_json(orient='records', lines=True)

# 将 JSON 数据写入文件
with open('train.json', 'w') as f:
    f.write(train_json)

with open('validation.json', 'w') as f:
    f.write(validation_json)

with open('test.json', 'w') as f:
    f.write(test_json)

print("转换完成")