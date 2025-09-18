import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Assume you already have:
# - MRIMultiSliceDataset class (from previous code)
# - MRISequenceClassifier class (from previous code)


def compute_accuracy(logits, labels):
    """Compute classification accuracy for multi-class"""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    total_acc_seq = 0
    total_acc_plane = 0
    total_acc_body = 0
    total_acc_contrast = 0
    total_samples = 0

    criterion_seq = torch.nn.CrossEntropyLoss()
    criterion_plane = torch.nn.CrossEntropyLoss()
    criterion_body = torch.nn.CrossEntropyLoss()
    criterion_contrast = torch.nn.BCEWithLogitsLoss()

    for images, metadata, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)  # (B, N_slices, C, H, W)
        metadata = metadata.to(device)  # (B, metadata_dim)

        seq_labels = labels["sequence"].to(device)
        plane_labels = labels["plane"].to(device)
        body_labels = labels["body"].to(device)
        contrast_labels = labels["contrast"].to(device)

        optimizer.zero_grad()

        seq_logits, plane_logits, body_logits, contrast_logits = model(images, metadata)

        loss_seq = criterion_seq(seq_logits, seq_labels)
        loss_plane = criterion_plane(plane_logits, plane_labels)
        loss_body = criterion_body(body_logits, body_labels)
        loss_contrast = criterion_contrast(contrast_logits.squeeze(), contrast_labels)

        loss = loss_seq + loss_plane + loss_body + loss_contrast
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # Compute accuracies
        total_acc_seq += compute_accuracy(seq_logits, seq_labels) * batch_size
        total_acc_plane += compute_accuracy(plane_logits, plane_labels) * batch_size
        total_acc_body += compute_accuracy(body_logits, body_labels) * batch_size

        # For contrast (binary), threshold at 0.5 sigmoid output
        contrast_preds = (torch.sigmoid(contrast_logits.squeeze()) > 0.5).long()
        total_acc_contrast += (contrast_preds == contrast_labels.long()).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc_seq = total_acc_seq / total_samples
    avg_acc_plane = total_acc_plane / total_samples
    avg_acc_body = total_acc_body / total_samples
    avg_acc_contrast = total_acc_contrast / total_samples

    return avg_loss, avg_acc_seq, avg_acc_plane, avg_acc_body, avg_acc_contrast


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    total_acc_seq = 0
    total_acc_plane = 0
    total_acc_body = 0
    total_acc_contrast = 0
    total_samples = 0

    criterion_seq = torch.nn.CrossEntropyLoss()
    criterion_plane = torch.nn.CrossEntropyLoss()
    criterion_body = torch.nn.CrossEntropyLoss()
    criterion_contrast = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, metadata, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            metadata = metadata.to(device)

            seq_labels = labels["sequence"].to(device)
            plane_labels = labels["plane"].to(device)
            body_labels = labels["body"].to(device)
            contrast_labels = labels["contrast"].to(device)

            seq_logits, plane_logits, body_logits, contrast_logits = model(
                images, metadata
            )

            loss_seq = criterion_seq(seq_logits, seq_labels)
            loss_plane = criterion_plane(plane_logits, plane_labels)
            loss_body = criterion_body(body_logits, body_labels)
            loss_contrast = criterion_contrast(
                contrast_logits.squeeze(), contrast_labels
            )

            loss = loss_seq + loss_plane + loss_body + loss_contrast
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            # Compute accuracies
            total_acc_seq += compute_accuracy(seq_logits, seq_labels) * batch_size
            total_acc_plane += compute_accuracy(plane_logits, plane_labels) * batch_size
            total_acc_body += compute_accuracy(body_logits, body_labels) * batch_size

            contrast_preds = (torch.sigmoid(contrast_logits.squeeze()) > 0.5).long()
            total_acc_contrast += (
                (contrast_preds == contrast_labels.long()).sum().item()
            )

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc_seq = total_acc_seq / total_samples
    avg_acc_plane = total_acc_plane / total_samples
    avg_acc_body = total_acc_body / total_samples
    avg_acc_contrast = total_acc_contrast / total_samples

    return avg_loss, avg_acc_seq, avg_acc_plane, avg_acc_body, avg_acc_contrast


def main_training_loop(
    train_dataset, val_dataset, num_epochs=10, batch_size=8, learning_rate=1e-4
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
        "sequence": 5,  # Example classes count
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mri_model.pth")
            print("Saved Best Model")


if __name__ == "__main__":
    # Example: create your train and val datasets here
    # train_dataset = MRIMultiSliceDataset(train_series_dirs, train_labels_dict, metadata_fields, num_slices=3)
    # val_dataset = MRIMultiSliceDataset(val_series_dirs, val_labels_dict, metadata_fields, num_slices=3)

    # main_training_loop(train_dataset, val_dataset, num_epochs=20, batch_size=8)
    pass
