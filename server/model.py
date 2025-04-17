import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Configuration
RANDOM_SEED = 42
FEATURE_DIM = 2  # x and y coordinates per landmark
BATCH_SIZE = 64
NUM_EPOCHS = 200  # Increased to allow for more exploration
LEARNING_RATE = 0.001
D_MODEL = 256     # Increased model capacity
NHEAD = 8         # More attention heads
NUM_LAYERS = 6    # Deeper architecture
DROPOUT = 0.3     # More regularization
NUM_CLASSES = 5   # Update with your actual number of classes
PATIENCE = 15     # More tolerant early stopping

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(RANDOM_SEED)

# ------------------------
# Enhanced Dataset Class
# ------------------------

class FacialLandmarksDataset(Dataset):
    def _init_(self, features, labels, is_training=False):
        self.features = features
        self.labels = labels
        self.is_training = is_training

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        landmark_seq = self.features[idx].reshape((-1, FEATURE_DIM))

        # Data augmentation
        if self.is_training:
            # Add random Gaussian noise
            noise = np.random.normal(0, 0.01, landmark_seq.shape)
            landmark_seq += noise

            # Random horizontal flip (assuming symmetric landmarks)
            if np.random.rand() > 0.5:
                landmark_seq[:, 0] = 1 - landmark_seq[:, 0]  # Flip x-coordinates

        return {
            'landmarks': torch.tensor(landmark_seq, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ------------------------
# Advanced Transformer Model
# ------------------------

class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=5000):
        super()._init_()
        self.dropout = nn.Dropout(DROPOUT)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EmotionTransformer(nn.Module):
    def _init_(self):
        super()._init_()
        self.embedding = nn.Sequential(
            nn.Linear(FEATURE_DIM, D_MODEL),
            nn.LayerNorm(D_MODEL)
        )
        self.pos_encoder = PositionalEncoding(D_MODEL)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT,
            dim_feedforward=D_MODEL*4,  # Larger feedforward network
            norm_first=True  # Better convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, NUM_LAYERS)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL//2, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average across sequence
        return self.classifier(x)

# ------------------------
# Training Infrastructure
# ------------------------

def prepare_loaders(df):
    X = df.drop('label', axis=1).values
    y = LabelEncoder().fit_transform(df['label'])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create datasets
    train_set = FacialLandmarksDataset(X_train, y_train, is_training=True)
    val_set = FacialLandmarksDataset(X_val, y_val)

    # Create loaders
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE*2, shuffle=False)

    return train_loader, val_loader, scaler

def train_model(train_loader, val_loader):
    model = EmotionTransformer().to(device)

    # Weighted loss for class imbalance
    class_weights = torch.tensor(
        np.bincount(train_loader.dataset.labels) / len(train_loader.dataset),
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=1/class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch['landmarks'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['landmarks'].to(device)
                labels = batch['label'].cpu().numpy()

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        # Record metrics
        history['train_loss'].append(epoch_loss / len(train_loader.dataset))
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1:03d}: '
              f'Train Loss: {history["train_loss"][-1]:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Best Acc: {best_acc:.4f}')

        # Early stopping
        if no_improve >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

# ------------------------
# Main Execution
# ------------------------

if __name__ == '_main_':
    # Load and prepare data
    df = pd.read_csv('/content/facial_landmarks.csv')
    SEQ_LENGTH = (len(df.columns) - 1) // FEATURE_DIM  # Auto-calculate

    train_loader, val_loader, scaler = prepare_loaders(df)

    # Train model
    model, history = train_model(train_loader, val_loader)

    # Save final artifacts
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'config': {
            'seq_length': SEQ_LENGTH,
            'feature_dim': FEATURE_DIM,
            'num_classes': NUM_CLASSES
        }
    }, 'emotion_transformer.pth')

    # Plot results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.show()