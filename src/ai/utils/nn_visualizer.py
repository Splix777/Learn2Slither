from typing import Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter

class NNVisualizer:
    def __init__(self, log_dir: str):
        """Initialize TensorBoard SummaryWriter for logging."""
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, scalar_value, global_step or self.global_step)

    def add_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """Visualize the model's graph."""
        self.writer.add_graph(model, input_to_model)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def step(self):
        """Increment the global step counter."""
        self.global_step += 1