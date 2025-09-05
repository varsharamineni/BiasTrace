import logging
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
from vllm import LLM

class vLLMClient:

    def __init__(self, model: str = "Qwen/Qwen3-4B", tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        # Suppress vLLM's tqdm progress bars
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

    def load_vllm(self):
        print(f"INFO:: Loading model: {self.model}")
        llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,  # Required for Qwen models
            max_model_len=32768,  # Qwen context length
            enable_prefix_caching=True,  # Enable prefix caching for better batching performance
            enforce_eager=False,  # Use CUDA graphs for better performance
            disable_log_stats=True,  # Disable vLLM's internal logging
        )
        print(f"INFO:: Model loaded successfully: {self.model}")
        return llm