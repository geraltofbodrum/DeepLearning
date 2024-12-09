import torch
import os
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

LATENT_DIR = "/home/aras/Desktop/University Folder/Deep Learning/SimOut/Latent Vectors"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_DIM = 512
NUM_CLASSES = 3  #Number of age groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_with_age_groups_from_filenames(latent_dir):
    print("Loading latent vectors and grouping ages...")
    latent_vectors = []
    age_groups = []

    def map_age_to_group(age):
        if age <= 10:
            return 0
        elif age <= 35:
            return 1
        else:
            return 2

    for filename in os.listdir(latent_dir):
        if filename.endswith(".pt"):
            #30_1_0_20170117094404388_latent.pt
            match = re.match(r"(\d+)_.*_latent\.pt", filename)
            if match:
                age = int(match.group(1))
                vector_path = os.path.join(latent_dir, filename)
                latent_vectors.append(torch.load(vector_path).squeeze(0).to(device))
                age_groups.append(map_age_to_group(age))

    latent_vectors = torch.stack(latent_vectors)
    age_groups = torch.tensor(age_groups).to(device)
    print(f"Loaded {len(latent_vectors)} samples.")
    return latent_vectors, age_groups

def split_data(latent_vectors, labels, test_size=0.2):
    print("Splitting data into training and testing sets...")
    return train_test_split(latent_vectors, labels, test_size=test_size, random_state=42)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
    print("Training the classifier...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for latent, label in train_loader:
            latent, label = latent.to(device), label.to(device)

            label = label.long()

            optimizer.zero_grad()
            output = model(latent)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_model_with_confusion_matrix(model, test_loader):
    print("Evaluating the classifier...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for latent, label in test_loader:
            latent, label = latent.to(device), label.to(device)

            label = label.long()

            output = model(latent)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")

    class_report = classification_report(all_labels, all_preds, target_names=["0-10", "11-35", "36-60"])
    print(f"Classification Report:\n{class_report}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0-10", "11-35", "36-60"], yticklabels=["0-10", "11-35", "36-60"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy, cm, class_report

if __name__ == "__main__":
    latent_vectors, age_groups = load_data_with_age_groups_from_filenames(LATENT_DIR)

    X_train, X_test, y_train, y_test = split_data(latent_vectors, age_groups)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)

    # Evaluate the model
    accuracy, cm, class_report = evaluate_model_with_confusion_matrix(model, test_loader)

    # Save classification report to a text file
    with open("classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(class_report)

