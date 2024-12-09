import os
import time
import pandas as pd
from rkllm_binding import *
import signal

MODEL_PATH = "/home/firefly/rkllm/qwen_half/Qwen_3576.rkllm"
handle = None

# 处理 Ctrl-C 退出
def signal_handler(signal, frame):
    print("Ctrl-C pressed, exiting...")
    global handle
    if handle:
        abort(handle)
        destroy(handle)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 设置日志级别
os.environ["RKLLM_LOG_LEVEL"] = "1"

inference_count = 0
inference_start_time = 0

def result_callback(result, userdata, state):
    global inference_start_time
    global inference_count
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        if inference_count == 0:
            first_token_time = time.time()
            print(f"首次生成令牌时间: {first_token_time - inference_start_time:.2f} 秒")
        inference_count += 1
        print(result.contents.text.decode(), end="", flush=True)
    elif state == LLMCallState.RKLLM_RUN_FINISH:
        print("\n\n(完成)")
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        print("\nLLM 调用过程中发生错误")

# 初始化 RKLLM
param = create_default_param()
param.model_path = MODEL_PATH.encode()
param.max_context_len = 1024
extend_param = RKLLMExtendParam()
extend_param.base_domain_id = 1
param.extend_param = extend_param
model_size = os.path.getsize(MODEL_PATH)
print(f"开始加载语言模型（大小: {model_size / 1024 / 1024:.2f} MB）")
start_time = time.time()
handle = init(param, result_callback)
end_time = time.time()
print(f"语言模型加载完成，用时 {end_time - start_time:.2f} 秒（速度: {model_size / (end_time - start_time) / 1024 / 1024:.2f} MB/s）")

# 加载数据集
train_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/train-00000-of-00001.parquet')
validation_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/validation-00000-of-00001.parquet')
test_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/test-00000-of-00001.parquet')

# 预处理数据
def preprocess_data(df):
    data = []
    for _, row in df.iterrows():
        question = row['question']
        choices = row['choices']['text']
        answer = row['answerKey']
        data.append((question, choices, answer))
    return data

validation_data = preprocess_data(validation_df)

# 模型推理
correct_predictions = 0
total_predictions = 0

for question, choices, correct_answer in validation_data:
    print("question : ", question)
    prompt = f"system\nYou are a helpful assistant.\nuser\n{question}\nassistant\n"
    for i, choice in enumerate(choices):
        prompt += f"{i + 1}. {choice}\n"
    prompt += "Please select the correct answer by number.\n"
    print("load done")

    # 创建 RKLLM 输入
    rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_PROMPT, prompt=prompt)

    # 创建推理参数
    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value

    # 运行 RKLLM
    inference_start_time = time.time()
    run(handle, rkllm_input, infer_param, None)

    # 获取模型输出
    # 假设 result_callback 会将结果存储在某个全局变量中
    # 这里需要根据实际情况修改
    model_output = get_model_output()  # 需要实现该函数以获取模型输出
    predicted_answer = parse_model_output(model_output)  # 需要实现该函数以解析模型输出

    if predicted_answer == correct_answer:
        correct_predictions += 1
    total_predictions += 1

# 计算精度
accuracy = correct_predictions / total_predictions
print(f"模型在验证集上的精度: {accuracy:.2f}")

# 清理资源
destroy(handle)