# import os
# import time
# from rkllm_binding import *
# import signal

# MODEL_PATH = "/home/firefly/rkllm/qwen_half/Qwen_3576.rkllm"
# handle = None

# # 处理 Ctrl-C 退出
# def signal_handler(signal, frame):
#     print("Ctrl-C pressed, exiting...")
#     global handle
#     if handle:
#         abort(handle)
#         destroy(handle)
#     exit(0)

# signal.signal(signal.SIGINT, signal_handler)

# # 设置日志级别
# os.environ["RKLLM_LOG_LEVEL"] = "1"

# inference_count = 0
# inference_start_time = 0

# def result_callback(result, userdata, state):
#     global inference_start_time
#     global inference_count
#     if state == LLMCallState.RKLLM_RUN_NORMAL:
#         if inference_count == 0:
#             first_token_time = time.time()
#             print(f"首次生成令牌时间: {first_token_time - inference_start_time:.2f} 秒")
#         inference_count += 1
#         print(result.contents.text.decode(), end="", flush=True)
#     elif state == LLMCallState.RKLLM_RUN_FINISH:
#         print("\n\n(完成)")
#     elif state == LLMCallState.RKLLM_RUN_ERROR:
#         print("\nLLM 调用过程中发生错误")

# # 初始化 RKLLM
# param = create_default_param()
# param.model_path = MODEL_PATH.encode()
# param.max_context_len = 1024
# extend_param = RKLLMExtendParam()
# extend_param.base_domain_id = 1
# param.extend_param = extend_param
# model_size = os.path.getsize(MODEL_PATH)
# print(f"开始加载语言模型（大小: {model_size / 1024 / 1024:.2f} MB）")
# start_time = time.time()
# handle = init(param, result_callback)
# end_time = time.time()
# print(f"语言模型加载完成，用时 {end_time - start_time:.2f} 秒（速度: {model_size / (end_time - start_time) / 1024 / 1024:.2f} MB/s）")

# print("开始推理...")

# # 创建输入提示
# prompt = """system
# You are a helpful assistant.
# user
# 你好，请告诉我今天的天气怎么样？
# assistant
# """

# # 创建 RKLLM 输入
# rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_PROMPT, prompt=prompt)

# # 创建推理参数
# infer_param = RKLLMInferParam()
# infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value

# # 运行 RKLLM
# inference_start_time = time.time()
# run(handle, rkllm_input, infer_param, None)

# # 清理资源
# destroy(handle)



import os
import time
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

print("开始对话...")

# 创建初始输入提示
prompt = """system
You are a helpful assistant.
"""

# while True:
if True:
    # user_input = input("user\n")

    user_input = "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
    prompt += f"user\n{user_input}\nassistant\n"

    # 创建 RKLLM 输入
    rkllm_input = create_rkllm_input(RKLLMInputType.RKLLM_INPUT_PROMPT, prompt=prompt)

    # 创建推理参数
    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE.value

    # 运行 RKLLM
    inference_start_time = time.time()
    run(handle, rkllm_input, infer_param, None)

# 清理资源
destroy(handle)