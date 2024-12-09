import os
import time
import pandas as pd
from rkllm_binding import *
import signal
import json

# MODEL_PATH = "/home/firefly/rkllm/qwen_half/Qwen_3576.rkllm"
MODEL_PATH = "/models/LLM/chatglm3-6b-3576-w4a16.rkllm"
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
# train_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/train-00000-of-00001.parquet')
# validation_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/validation-00000-of-00001.parquet')
# test_df = pd.read_parquet('/userdata/repos/datasets/commonsense_qa/data/test-00000-of-00001.parquet')

val_datasets_path = '/userdata/repos/datasets/gsm8k/main/test-00000-of-00001.parquet'
validation_df = pd.read_parquet(val_datasets_path)

# 预处理数据
def preprocess_data_commonsense_qa(df):
    data = []
    for _, row in df.iterrows():
        question = row['question']
        choices = row['choices']['text']
        answer = row['answerKey']
        data.append((question, choices, answer))
    return data

def preprocess_data_gsm8k(df):
    data = []
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        data.append((question, answer))
    return data

def extract_number_from_answer(text):
    """从答案文本中提取数字"""
    try:
        # 尝试从 #### 格式中提取
        if "####" in text:
            return float(text.split("####")[-1].strip().split()[0])
        
        # 如果没有 ####，尝试提取最后一个数字
        import re
        numbers = re.findall(r'-?\d*\.?\d+', text)
        if numbers:
            return float(numbers[-1])
        
        raise ValueError("未找到数字答案")
    except Exception as e:
        raise ValueError(f"提取数字失败: {str(e)}")

def evaluate_single_problem(handle, question, answer):
    """评估单个问题"""
    global inference_start_time, inference_count, model_output
    
    print("\n问题：", question)
    print("标准答案：", answer)
    
    # 构建 prompt
    prompt = f"""system
You are a helpful assistant that is good at solving math problems. Please solve the problem step by step.
user
{question}
assistant
Let me solve this step by step:
"""
    
    # 创建 RKLLM 输入和参数
    rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_PROMPT, prompt=prompt)
    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value
    infer_param.max_length = 1024
    infer_param.top_p = 0.8
    infer_param.temperature = 0.8

    # 重置模型输出
    model_output = ""
    inference_start_time = time.time()
    inference_count = 0
    
    def custom_callback(result, userdata, state):
        global model_output
        if state == LLMCallState.RKLLM_RUN_NORMAL:
            text = result.contents.text.decode()
            model_output += text
            print(text, end="", flush=True)
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            print("\n(完成)")
    
    # 运行模型
    run(handle, rkllm_input, infer_param, None)
    
    try:
        predicted_number = extract_number_from_answer(model_output)
        correct_number = extract_number_from_answer(answer)
        
        is_correct = abs(predicted_number - correct_number) < 1e-6
        print(f"预测答案：{predicted_number}")
        print(f"是否正确：{'✓' if is_correct else '✗'}")
        
        return is_correct, True  # (是否正确, 是否成功评估)
    except ValueError as e:
        print(f"评估失败：{e}")
        return False, False  # (是否正确, 是否成功评估)
def save_quant_data(validation_data, output_file="quant.json", num_samples=20):
    """将验证数据保存为量化所需的格式"""
    quant_data = []
    
    for i, (question, answer) in enumerate(validation_data):
        if i >= num_samples:
            break
            
        # 构建输入输出对
        data_item = {
            "input": f"Human: {question}\nAssistant: ",
            "target": answer
        }
        quant_data.append(data_item)
    
    # 保存到JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(quant_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存{len(quant_data)}条数据到{output_file}")

def run_evaluation(handle, validation_data):
    """运行整体评估"""
    correct_predictions = 0
    total_predictions = 0
    failed_evaluations = 0
    
    for question, answer in validation_data:
        is_correct, is_evaluated = evaluate_single_problem(handle, question, answer)
        
        if is_evaluated:
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
        else:
            failed_evaluations += 1
    
    # 输出评估结果
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n评估结果:")
        print(f"总样本数: {len(validation_data)}")
        print(f"成功评估数: {total_predictions}")
        print(f"评估失败数: {failed_evaluations}")
        print(f"正确预测数: {correct_predictions}")
        print(f"模型精度: {accuracy:.2%}")
    else:
        print("\n没有成功评估任何样本")

# 主执行流程
validation_data = preprocess_data_gsm8k(validation_df)
run_evaluation(handle, validation_data)

# 保存量化数据
# save_quant_data(validation_data)


# 清理资源
destroy(handle)