import ctypes
import numpy as np
from enum import IntEnum
from typing import Callable, Any

# Load the shared library
_lib = ctypes.CDLL("./librkllmrt.so")  # Adjust the library name if necessary

# Define enums
class LLMCallState(IntEnum):
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3
    RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4

class RKLLMInputType(IntEnum):
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3

class RKLLMInferMode(IntEnum):
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

# Define structures
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 112)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam)
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t)
    ]

class RKLLMInput(ctypes.Structure):
    class _InputUnion(ctypes.Union):
        _fields_ = [
            ("prompt_input", ctypes.c_char_p),
            ("embed_input", RKLLMEmbedInput),
            ("token_input", RKLLMTokenInput),
            ("multimodal_input", RKLLMMultiModelInput)
        ]
    _fields_ = [
        ("input_type", ctypes.c_int),
        ("_input", _InputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam))
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer)
    ]

# Define callback type
LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# Define function prototypes
_lib.rkllm_createDefaultParam.restype = RKLLMParam
_lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]
_lib.rkllm_init.restype = ctypes.c_int
_lib.rkllm_load_lora.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMLoraAdapter)]
_lib.rkllm_load_lora.restype = ctypes.c_int
_lib.rkllm_load_prompt_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.rkllm_load_prompt_cache.restype = ctypes.c_int
_lib.rkllm_release_prompt_cache.argtypes = [ctypes.c_void_p]
_lib.rkllm_release_prompt_cache.restype = ctypes.c_int
_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
_lib.rkllm_destroy.restype = ctypes.c_int
_lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
_lib.rkllm_run.restype = ctypes.c_int
_lib.rkllm_run_async.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
_lib.rkllm_run_async.restype = ctypes.c_int
_lib.rkllm_abort.argtypes = [ctypes.c_void_p]
_lib.rkllm_abort.restype = ctypes.c_int
_lib.rkllm_is_running.argtypes = [ctypes.c_void_p]
_lib.rkllm_is_running.restype = ctypes.c_int

# Python wrapper functions
def create_default_param() -> RKLLMParam:
    return _lib.rkllm_createDefaultParam()

def init(param: RKLLMParam, callback: Callable[[RKLLMResult, Any, LLMCallState], None]) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    c_callback = LLMResultCallback(callback)
    status = _lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), c_callback)
    if status != 0:
        raise RuntimeError(f"Failed to initialize RKLLM: {status}")
    return handle

def load_lora(handle: ctypes.c_void_p, lora_adapter: RKLLMLoraAdapter) -> None:
    status = _lib.rkllm_load_lora(handle, ctypes.byref(lora_adapter))
    if status != 0:
        raise RuntimeError(f"Failed to load Lora adapter: {status}")

def load_prompt_cache(handle: ctypes.c_void_p, prompt_cache_path: str) -> None:
    status = _lib.rkllm_load_prompt_cache(handle, prompt_cache_path.encode())
    if status != 0:
        raise RuntimeError(f"Failed to load prompt cache: {status}")

def release_prompt_cache(handle: ctypes.c_void_p) -> None:
    status = _lib.rkllm_release_prompt_cache(handle)
    if status != 0:
        raise RuntimeError(f"Failed to release prompt cache: {status}")

def destroy(handle: ctypes.c_void_p) -> None:
    status = _lib.rkllm_destroy(handle)
    if status != 0:
        raise RuntimeError(f"Failed to destroy RKLLM: {status}")

def run(handle: ctypes.c_void_p, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata: Any) -> None:
    status = _lib.rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), ctypes.c_void_p(userdata))
    if status != 0:
        raise RuntimeError(f"Failed to run RKLLM: {status}")

def run_async(handle: ctypes.c_void_p, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata: Any) -> None:
    status = _lib.rkllm_run_async(handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), ctypes.c_void_p(userdata))
    if status != 0:
        raise RuntimeError(f"Failed to run RKLLM asynchronously: {status}")

def abort(handle: ctypes.c_void_p) -> None:
    status = _lib.rkllm_abort(handle)
    if status != 0:
        raise RuntimeError(f"Failed to abort RKLLM: {status}")

def is_running(handle: ctypes.c_void_p) -> bool:
    return _lib.rkllm_is_running(handle) == 0

# Helper function to convert numpy array to C array
def numpy_to_c_array(arr: np.ndarray, c_type):
    return arr.ctypes.data_as(ctypes.POINTER(c_type))

# Helper function to create RKLLMInput
def create_rkllm_input(input_type: RKLLMInputType, **kwargs) -> RKLLMInput:
    rkllm_input = RKLLMInput()
    rkllm_input.input_type = input_type.value

    if input_type == RKLLMInputType.RKLLM_INPUT_PROMPT:
        rkllm_input._input.prompt_input = kwargs['prompt'].encode()
    elif input_type == RKLLMInputType.RKLLM_INPUT_EMBED:
        embed = kwargs['embed']
        rkllm_input._input.embed_input.embed = numpy_to_c_array(embed, ctypes.c_float)
        rkllm_input._input.embed_input.n_tokens = embed.shape[1]
    elif input_type == RKLLMInputType.RKLLM_INPUT_TOKEN:
        tokens = kwargs['tokens']
        rkllm_input._input.token_input.input_ids = numpy_to_c_array(tokens, ctypes.c_int32)
        rkllm_input._input.token_input.n_tokens = tokens.shape[1]
    elif input_type == RKLLMInputType.RKLLM_INPUT_MULTIMODAL:
        rkllm_input._input.multimodal_input.prompt = kwargs['prompt'].encode()
        image_embed = kwargs['image_embed']
        rkllm_input._input.multimodal_input.image_embed = numpy_to_c_array(image_embed, ctypes.c_float)
        rkllm_input._input.multimodal_input.n_image_tokens = image_embed.shape[1]

    return rkllm_input