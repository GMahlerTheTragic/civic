import os
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel


class GpuDeviceHandler:
    @staticmethod
    def use_cuda_if_available():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is unavailable")
        print(f"Using device: {device}")
        return device

    @staticmethod
    def use_data_parallelism_if_available(model):
        is_distributed = torch.cuda.device_count() > 1
        if is_distributed:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        return model

    @staticmethod
    def setup_distributed_data_parallelism(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group("gloo", rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        destroy_process_group()

    @staticmethod
    def use_distributed_data_parallelism_if_available(model, rank, world_size):
        is_distributed = torch.cuda.device_count() > 1
        if is_distributed:
            GpuDeviceHandler.setup_distributed_data_parallelism(rank, world_size)
            model = model().to(rank)
            model = DistributedDataParallel(model, device_ids=[rank])
        return model

    @staticmethod
    def is_model_using_data_parallel(model):
        if isinstance(model, nn.DataParallel) or isinstance(
            model, nn.parallel.DistributedDataParallel
        ):
            return True
        else:
            return False
