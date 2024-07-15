from typing import Any
import gc, os
import psutil
import pynvml

import torch
import tempfile


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

def show_memory():
    used_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
    print(f"Used GPU Memory: {used_memory:.2f} GB")

    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # cpu_memory = psutil.virtual_memory()
    # cpu_total = cpu_memory.total/1024**3
    # cpu_used = cpu_memory.used/1024**3
    
    # cuda_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # cuda_total = cuda_info.total/1024**2
    # cuda_used = cuda_info.used/1024**2
    
    # torch_cuda_allocated = torch.cuda.memory_allocated()/1024**2 
    # torch_cuda_reserved = torch.cuda.memory_reserved()/1024**2 
    
    # print(f"\n**************************** Current Memory ****************************\n")
    # print(f"    Total CPU Memory: {cpu_total:10.2f} GB | Used CPU Memory: {cpu_used:10.1f} GB")
    # print(f"    -----------------------------------------------------------------")
    # print(f"    Total GPU Memory: {cuda_total:10.2f} MB | Used GPU Memory: {cuda_used:10.2f} MB")
    # print(f"    -----------------------------------------------------------------")
    # print(f"    Torch_Cuda_Alloc: {torch_cuda_allocated:10.2f} MB | Torch_Cuda_Rsrv: {torch_cuda_reserved:10.2f} MB")
    # print(f"\n**************************** Current Memory ****************************\n")
    
    # pynvml.nvmlShutdown()
    # return cpu_used, cuda_used

class Memory:
    def __init__(self, name, value, desc, path) -> None:
        self.name = name
        self.desc = desc
        self.path = path
        if desc == 'cpu':
            self._value = value.cpu()            
        elif desc.startswith('cuda'):
            self._value = value.to(desc)
        elif desc == 'disk':
            torch.save(value, self.path)
            self._value = None
            value = value.cpu()
            del value
        clear_mem()
    
    def to(self, desc='cpu'):
        if desc == self.desc:
            return self

        if self.desc == 'disk':
            self._value = torch.load(self.path)

        if desc.startswith('cuda'):
            self._value = self._value.to(desc)
        
        elif desc == 'cpu':
            self._value = self._value.cpu()
        
        elif desc == 'disk':
            torch.save(self._value.cpu(), self.path)
            del self._value
            self._value = None
        
        else:
            raise RuntimeError('cannot put to', desc)

        self.desc = desc
        clear_mem()
        return self
    
    def __call__(self) -> Any:
        if self.desc == 'disk':
            res = torch.load(self.path, map_location='cpu')
            self._value = res
            self.desc = 'cpu'
        return self._value

    @property
    def value(self):
        return self()
    
    @value.setter
    def value(self, value):
        assert isinstance(value, torch.Tensor), 'currently only support torch tensor'
        if self.desc.startswith('cuda'):
            self._value = value.to(self.desc)
        elif self.desc == 'cpu':
            self._value = value.cpu()
        elif self.desc == 'disk':
            torch.save(value.cpu(), self.path)
        else:
            raise RuntimeError('cannot put to', self.desc)

    def __del__(self):
        del self._value
        clear_mem()

class MemoryBank:
    def __init__(self, disk_path=None) -> None:
        self.record = {}
        if disk_path is None:
            __temp_dir = tempfile.TemporaryDirectory()
            disk_path = __temp_dir.name
        self.disk_path = disk_path
    
    def add_value(self, key, value, desc='cpu') -> Memory:
        self.record[key] = Memory(key, value, desc, os.path.join(self.disk_path, key + '.pth'))
        return self.record[key]

    def del_value(self, key):
        del self.record[key]
        clear_mem()
    
    def __del__(self):
        keys = list(self.record.keys())
        for key in keys:
            self.del_value(key)
        del self.record
        clear_mem()


__temp_dir = tempfile.TemporaryDirectory()
__temp_dir_path = __temp_dir.name

MEMORY_BANK = MemoryBank(__temp_dir_path)


