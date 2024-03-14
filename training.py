import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained SciBERT tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

# Load your training data from an Excel file
df = pd.read_excel('data/train_sentences.xlsx')

# Shuffle the DataFrame to mix positive and negative examples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Tokenize the paragraphs and convert to PyTorch tensors
tokenized = tokenizer(df['sentence'].tolist(), padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(df['label'].tolist())

# Create a DataLoader for handling batches during training
dataset = TensorDataset(tokenized.input_ids, tokenized.attention_mask, labels)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set up the optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    
    for batch in train_dataloader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        # Calculate the loss using logits and labels
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()

# Save the trained model
model.save_pretrained("scibert-model")