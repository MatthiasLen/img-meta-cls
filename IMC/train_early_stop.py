import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


# Training and validation functions remain the same as before (train_epoch, validate_epoch)


def main_training_loop(
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    patience=7,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num_classes_dict = {
        "sequence": 5,  # adjust as per your dataset
        "plane": 3,
        "body": 5,
        "contrast": 1,
    }

    metadata_dim = train_dataset[0][1].shape[0]

    model = MRISequenceClassifier(
        metadata_input_dim=metadata_dim, num_classes_dict=num_classes_dict
    )
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        (
            train_loss,
            train_acc_seq,
            train_acc_plane,
            train_acc_body,
            train_acc_contrast,
        ) = train_epoch(model, train_loader, optimizer, device)
        print(
            f"Train Loss: {train_loss:.4f} | Seq Acc: {train_acc_seq:.4f} | Plane Acc: {train_acc_plane:.4f} | Body Acc: {train_acc_body:.4f} | Contrast Acc: {train_acc_contrast:.4f}"
        )

        val_loss, val_acc_seq, val_acc_plane, val_acc_body, val_acc_contrast = (
            validate_epoch(model, val_loader, device)
        )
        print(
            f"Val   Loss: {val_loss:.4f} | Seq Acc: {val_acc_seq:.4f} | Plane Acc: {val_acc_plane:.4f} | Body Acc: {val_acc_body:.4f} | Contrast Acc: {val_acc_contrast:.4f}"
        )

        # Step the scheduler with validation loss
        scheduler.step(val_loss)

        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mri_model.pth")
            print("Saved Best Model")

        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break


if __name__ == "__main__":
    # Prepare your datasets and call main_training_loop(train_dataset, val_dataset, ...)
    pass
