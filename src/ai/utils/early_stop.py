import torch

class EarlyStop:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """Initialize the EarlyStop object for model performance."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss, model) -> None:
        """Check if training should be stopped based on the loss."""
        if self.best_loss is None:
            self.best_loss = current_loss
            # self.save_checkpoint(model)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            # self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save the model's current state."""
        torch.save(model.state_dict(), 'checkpoint.pth')
