import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        # If multiple CUDA devices are available, select the one with most free memory
        device_count = torch.cuda.device_count()
        if device_count > 1:
            max_free_memory = 0
            best_device = 0
            for i in range(device_count):
                free_memory, total_memory = torch.cuda.mem_get_info(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i
            return f"cuda:{best_device}"
        else:
            return "cuda"
    else:
        return "cpu"
