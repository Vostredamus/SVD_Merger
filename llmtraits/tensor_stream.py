import math
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from .paths import relocated_path


DEFAULT_CHUNK_SIZE = 1_048_576


def tensor_numel(shape) -> int:
    return int(math.prod(shape)) if shape else 1


def iter_flat_ranges(shape, chunk_size: int = DEFAULT_CHUNK_SIZE):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = tensor_numel(shape)
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def read_flat_slice(handle, name: str, shape, start: int, end: int) -> np.ndarray:
    shape = tuple(int(value) for value in shape)
    total = tensor_numel(shape)
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid flat range [{start}:{end}) for {name} with {total} values")
    if start == end:
        return np.empty(0, dtype=np.float32)
    if not shape:
        raise ValueError(f"Scalar tensor streaming is not supported for {name}")

    tensor_slice = handle.get_slice(name)
    if tuple(tensor_slice.get_shape()) != shape:
        raise ValueError(f"Shape changed for {name}: expected {shape}, got {tensor_slice.get_shape()}")

    if len(shape) == 1:
        value = tensor_slice[start:end]
        return value.to(torch.float32).contiguous().numpy().reshape(-1)

    inner_size = int(math.prod(shape[1:]))
    first_row = start // inner_size
    final_row = (end + inner_size - 1) // inner_size
    value = tensor_slice[first_row:final_row]
    flat = value.to(torch.float32).contiguous().numpy().reshape(-1)
    local_start = start - first_row * inner_size
    return flat[local_start : local_start + (end - start)]


@contextmanager
def open_model_tensor_slices(models: list[dict], name: str):
    with ExitStack() as stack:
        handles = []
        shape = None
        for model in models:
            record = model["manifest"].tensors[name]
            current_shape = tuple(record.shape)
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise ValueError(f"Shape mismatch for shared tensor {name}")
            shard_path = relocated_path(model["delta_dir"]) / record.shard_file
            handles.append(
                stack.enter_context(safe_open(str(shard_path), framework="pt", device="cpu"))
            )
        yield handles, shape


def streamed_tensor_gram(models: list[dict], name: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
    n_models = len(models)
    gram = np.zeros((n_models, n_models), dtype=np.float64)
    shape = tuple(models[0]["manifest"].tensors[name].shape)
    for model in models[1:]:
        if tuple(model["manifest"].tensors[name].shape) != shape:
            raise ValueError(f"Shape mismatch for shared tensor {name}")
    for start, end in iter_flat_ranges(shape, chunk_size):
        block = np.empty((n_models, end - start), dtype=np.float32)
        for model_index, model in enumerate(models):
            record = model["manifest"].tensors[name]
            shard_path = relocated_path(model["delta_dir"]) / record.shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                block[model_index] = read_flat_slice(handle, name, shape, start, end)
        block64 = block.astype(np.float64)
        gram += block64 @ block64.T
    return gram
