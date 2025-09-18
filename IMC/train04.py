import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Helper: initialize weights
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# Example multi-task loss combining all heads
class MultiTaskLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        seq_logits, plane_logits, body_logits, contrast_logits = preds
        seq_t, plane_t, body_t, contrast_t = targets

        loss_seq = self.ce_loss(seq_logits, seq_t)
        loss_plane = self.ce_loss(plane_logits, plane_t)
        loss_body = self.ce_loss(body_logits, body_t)
        loss_contrast = self.bce_loss(contrast_logits.squeeze(), contrast_t.float())

        total_loss = loss_seq + loss_plane + loss_body + loss_contrast
        return total_loss, (loss_seq.item(), loss_plane.item(), loss_body.item(), loss_contrast.item())

# Warmup + cosine scheduler helper
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + torch.cos(torch.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_optimizer(model, lr=1e-4, weight_decay=1e-4):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "norm", "ln", "layernorm"]):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )

def train_loop(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    save_path,
    warmup_steps=1000,
    total_steps=10000,
):
    model.to(device)
    model.apply(init_weights)

    optimizer = create_optimizer(model)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion = MultiTaskLoss(label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float("inf")
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, metadata, targets = batch
            images = images.to(device)
            metadata = metadata.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images, metadata)
                loss, _ = criterion(outputs, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, metadata, targets = batch
                images = images.to(device)
                metadata = metadata.to(device)
                targets = [t.to(device) for t in targets]

                with torch.cuda.amp.autocast():
                    outputs = model(images, metadata)
                    loss, _ = criterion(outputs, targets)

                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Early stopping & checkpointing
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")


# Example usage skeleton:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assume you have your dataset and dataloaders ready: train_loader, val_loader
    # Each batch from dataloader returns: images (B,N,C,H,W), metadata (B,meta_dim), targets = (seq_t, plane_t, body_t, contrast_t)

    model = MRISequenceClassifier(metadata_input_dim=512*3, num_classes_dict={
        "sequence": 5, "plane": 3, "body": 5, "contrast": 1
    })

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        device=device,
        save_path="best_model.pth",
        warmup_steps=500,
        total_steps=10000,
    )
