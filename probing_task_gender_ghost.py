import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

PKL_FILES = ["log_1.pkl", "log_1_1.pkl", "log_1_old.pkl"]
SAMPLE_SIZE = 500
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005
INPUT_DIM = 512
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_pkls(pkl_files, sample_size):

    print("Loading data from .pkl files...")
    latent_vectors = []
    genders = []

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        sampled_data = random.sample(data, min(sample_size, len(data)))

        for sublist in sampled_data:
            for entry in sublist:
                input_info = entry['input_info']
                identity = entry['target_identity']

                if isinstance(identity, list):
                    identity = torch.stack([torch.tensor(elem) for elem in identity])

                if identity.shape[0] > 1:
                    identity_tensor = identity[0].to(device)
                else:
                    identity_tensor = identity.squeeze(0).to(device)

                identity_tensor = identity_tensor.view(1, -1)

                gender = int(input_info.split('_')[1])
                latent_vectors.append(identity_tensor)
                genders.append(gender)

    latent_vectors = torch.cat(latent_vectors, dim=0)
    genders = torch.tensor(genders).to(device)
    print(f"Loaded {len(latent_vectors)} samples from {len(pkl_files)} files.")
    return latent_vectors, genders

def split_data(latent_vectors, labels, test_size=0.2):
    print("Splitting data into training and testing sets...")
    return train_test_split(latent_vectors, labels, test_size=test_size, random_state=42)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


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
                label = label.long()
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
    plt.savefig("loss_curve_gender.png")

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

    class_report = classification_report(all_labels, all_preds, target_names=["Male", "Female"])
    print(f"Classification Report:\n{class_report}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix_gender.png")
    print("Confusion matrix saved as 'confusion_matrix_gender.png'.")

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, cm, class_report

if __name__ == "__main__":
    latent_vectors, genders = load_data_from_pkls(PKL_FILES, SAMPLE_SIZE)
    X_train, X_temp, y_train, y_temp = split_data(latent_vectors, genders)
    X_val, X_test, y_val, y_test = split_data(X_temp, y_temp, test_size=0.5)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    accuracy, cm, class_report = evaluate_model_with_confusion_matrix(model, test_loader)

    with open("classification_report_gender.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(class_report)
    print("Classification report saved as 'classification_report_gender.txt'.")
