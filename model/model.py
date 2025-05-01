import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the VisionIQ architecture
class VisionIQ(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=3129):  # TextVQA has 3129 answers
        super(VisionIQ, self).__init__()

        # Visual feature extractor (CNN - ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC
        self.fc_img = nn.Linear(resnet.fc.in_features, hidden_dim)

        # Text encoder (LSTM with GloVe-style embeddings)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True)

        # Multimodal Transformer Fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, input_ids, attention_mask):
        # CNN
        img_feat = self.cnn(images).squeeze()
        img_feat = self.fc_img(img_feat).unsqueeze(1)  # (B, 1, H)

        # LSTM
        text_feat = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(text_feat)  # (B, T, H)

        # Concatenate and pass through transformer
        fused = torch.cat([img_feat, lstm_out], dim=1)
        fused = self.transformer(fused)
        output = self.classifier(fused[:, 0])  # Use first token for classification

        return output


# TextVQA Dataset loading and preprocessing
class TextVQADataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("textvqa", split=split)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(image)

        # Question
        question = entry["question"]
        encoding = tokenizer(question, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Answer (use first correct answer)
        label = entry["answers"]["text"][0]
        label_id = answer_vocab.get(label, answer_vocab["<unk>"])

        return image, input_ids, attention_mask, label_id


# Build answer vocabulary
def build_answer_vocab(dataset, threshold=5):
    from collections import Counter
    counter = Counter()
    for entry in dataset:
        counter.update(entry["answers"]["text"])
    vocab = {ans: i for i, (ans, count) in enumerate(counter.items()) if count >= threshold}
    vocab["<unk>"] = len(vocab)
    return vocab


# Load datasets
train_raw = load_dataset("textvqa", split="train[:5000]")
answer_vocab = build_answer_vocab(train_raw)
num_classes = len(answer_vocab)

# Initialize model
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
model = VisionIQ(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Dataloaders
train_dataset = TextVQADataset("train[:5000]")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
def train_model(epochs=100):
    model.train()
    for epoch in range(epochs):
        total, correct = 0, 0
        for images, input_ids, attention_mask, labels in train_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

train_model()

