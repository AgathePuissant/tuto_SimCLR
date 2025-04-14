import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, cohen_kappa_score
import argparse, os, time, copy, csv
from tqdm import tqdm
import pandas as pd
import csv
import fcntl  # For safe file locking on Linux systems

def safe_append_to_csv(row, csv_path, header):
    with open(csv_path, 'a+', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        write_header = f.tell() == 0 or not any(row.keys() == h for h in csv.DictReader(f))
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


# SimCLR Encoder Model
class Encoder(nn.Module):
    def __init__(self, base_model=models.resnet50, out_dim=128):
        super().__init__()
        self.encoder = base_model(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)

# SimCLR transform
def get_simclr_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Validation transform
def get_validation_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# NT-Xent Loss
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    similarity = torch.mm(z, z.T) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    labels = (labels + z.size(0) // 2) % z.size(0)
    return nn.CrossEntropyLoss()(similarity, labels)

# Train SimCLR model
def train_simclr(args):
    transform = get_simclr_transform()
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    # Weighted sampler
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    class_weights = [1.0 / c for c in class_counts]
    weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    model = Encoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training with temperature={args.temperature}, batch_size={args.batch_size}, epochs={args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for (x, _), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x1, x2 = transform(x), transform(x)
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), args.output_model_path)
    print(f"Model saved to {args.output_model_path}")
    return model.encoder

# Train linear classifier on frozen features
def train_linear_classifier(encoder, dataset_path, batch_size, device):
    transform = get_validation_transform()
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    class_names = dataset.classes

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Linear classifier model
    model = nn.Sequential(
        encoder,
        nn.Identity(),  # projection head removed
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, len(class_names))
    ).to(device)

    # Freeze encoder
    for param in model[0].parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(50):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            loader = train_loader if phase == 'train' else val_loader
            running_loss, running_corrects = 0.0, 0
            all_labels, all_preds = [], []

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / len(loader.dataset)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    return accuracy, f1, kappa

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--metrics_csv", default="simclr_validation_results.csv")
    args = parser.parse_args()

    encoder = train_simclr(args)
    acc, f1, kappa = train_linear_classifier(encoder, args.data_path, args.batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    header = ['model_name', 'temperature', 'batch_size', 'epochs', 'val_accuracy', 'val_f1', 'val_kappa']
    row = {
        'model_name': os.path.basename(args.output_model_path),
        'temperature': args.temperature,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'val_accuracy': acc,
        'val_f1': f1,
        'val_kappa': kappa
    }
    safe_append_to_csv(row, args.metrics_csv, header)

    print(f"Validation: Acc={acc:.2f}%, F1={f1:.4f}, Kappa={kappa:.4f}")
