import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os

# Use CPU as the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a smaller RNNTextGenerator class with reduced dimensions
class SmallRNNTextGenerator(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)  # Use batch_first=True
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        predictions = self.fc(output)
        return predictions

# Define a custom dataset for text data
class TextDataset(Dataset):
    def __init__(self, data_path, vocab):
        self.vocab = vocab
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = file.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # Tokenize and convert text to tensor using the vocabulary
        tokens = [self.vocab[word] for word in text.split()]
        tensor = torch.LongTensor(tokens)
        return tensor

# Hyperparameters with reduced dimensions
EMBEDDING_DIM = 16  # Smaller embedding dimension
HIDDEN_DIM = 32  # Smaller hidden dimension
OUTPUT_DIM = 10000
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2  # Gradient accumulation steps

# Load data and create vocabulary
data_path = 'tmp_execution/preprocessed_data.txt'

# Initialize an empty vocabulary
vocab = {}
index = 0

# Find the maximum token index in the preprocessed data
max_token_index = 0

with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        tokens = line.strip().split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
                max_token_index = max(max_token_index, vocab[token])

# Set input_dim to be larger than or equal to the maximum token index
input_dim = max_token_index + 1  # Add 1 to account for the zero-based index

# Initialize the smaller model, optimizer, and loss function
model = SmallRNNTextGenerator(input_dim, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Create dataset using the vocabulary
dataset = TextDataset(data_path, vocab)

# Modify collate_fn to pad sequences
def collate_batch(batch):
    return pad_sequence(batch, batch_first=True)

# Create data loader with the modified collate_fn and reduced batch size
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Training loop with memory optimization
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions.view(-1, OUTPUT_DIM), batch.view(-1))
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            total_loss += loss.item()
            del predictions, loss
            torch.cuda.empty_cache()

    # Perform the final optimization step if needed
    if i % ACCUMULATION_STEPS != 0:
        optimizer.step()
        total_loss += loss.item()
        del predictions, loss
        torch.cuda.empty_cache()

    # Print or log the average loss for the epoch
    average_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {average_loss:.4f}")

# Save trained model weights
output_directory = 'output'
os.makedirs(output_directory, exist_ok=True)
model_weights_path = os.path.join(output_directory, 'text_generator_model.pt')
torch.save(model.state_dict(), model_weights_path)
print(f"Trained model weights saved to {model_weights_path}")
