import torch
import os
import re
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

LATENT_DIR = "/home/aras/Desktop/University Folder/Deep Learning/SimOut/Latent Vectors"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_DIM = 512
NUM_CLASSES = 2  #Number of classes (gender: Male/Female)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latent_vectors(latent_dir):
    print("Loading latent vectors and extracting labels...")
    latent_vectors = []
    labels = []

    #30_1_0_20170117094404388_latent.pt
    filename_pattern = re.compile(r"(\d+)_(\d+)_.*_latent\.pt")

    for file in os.listdir(latent_dir):
        if file.endswith(".pt"):
            match = filename_pattern.match(file)
            if match:
                gender = int(match.group(2))  #Gender: 0 (male), 1 (female)

                latent_vector_path = os.path.join(latent_dir, file)
                latent_vector = torch.load(latent_vector_path).squeeze(0).to(device)

                latent_vectors.append(latent_vector)
                labels.append(gender)

    latent_vectors = torch.stack(latent_vectors)
    labels = torch.tensor(labels).to(device)

    print(f"Loaded {len(latent_vectors)} latent vectors.")
    return latent_vectors, labels

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    print("Training the classifier...")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for latent, label in train_loader:
            latent, label = latent.to(device), label.to(device)
            label = label.long()

            optimizer.zero_grad()
            output = model(latent)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for latent, label in val_loader:
                latent, label = latent.to(device), label.to(device)
                output = model(latent)
                loss = criterion(output, label)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Loss curve saved as 'loss_curve.png'.")

def evaluate_model(model, test_loader):
    print("Evaluating the classifier...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for latent, label in test_loader:
            latent, label = latent.to(device), label.to(device)
            output = model(latent)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    report = classification_report(all_labels, all_preds, target_names=["Female", "Male"])
    print("Classification Report:\n", report)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'.")

    return accuracy, cm, report

if __name__ == "__main__":
    latent_vectors, labels = load_latent_vectors(LATENT_DIR)

    X_train, X_test, y_train, y_test = train_test_split(latent_vectors, labels, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)

    evaluate_model(model, test_loader)
