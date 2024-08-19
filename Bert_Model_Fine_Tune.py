import json
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
import torch.cuda.amp as amp
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paths to locally stored model and tokenizer
MODEL_PATH = '/gpfswork/rech/fmr/uft12cr/finetuneAli/Bert_Model'
TOKENIZER_PATH = '/gpfswork/rech/fmr/uft12cr/finetuneAli/Bert_tokenizer'
General_TOKENIZER_PATH = '/gpfswork/rech/fmr/uft12cr/finetuneAli/vocab2.txt'

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
decoder = json.JSONDecoder()
data = []
with open('/gpfswork/rech/fmr/uft12cr/finetuneAli/arxiv-metadata-oai-snapshot.json') as f:
    for line in f:
        data.append(decoder.decode(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess abstracts
df['preprocessed_abstract'] = df['abstract'].apply(preprocess_text)

# Define domain-wise categories
domain_wise_categories = {
    "Mathematics": ["math.AG", "math.AT", "math.AP", "math.CT", "math.CA", "math.CO", "math.AC", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GT", "math.GR", "math.HO", "math.IT", "math.KT", "math.LO", "math.MP", "math.MG", "math.NT", "math.NA", "math.OA", "math.OC", "math.PR", "math.QA", "math.RT", "math.RA", "math.SP", "math.ST", "math.SG"],
    "Computer Science": ["cs.AI", "cs.CL", "cs.CC", "cs.CE", "cs.CG", "cs.GT", "cs.CV", "cs.CY", "cs.CR", "cs.DS", "cs.DB", "cs.DL", "cs.DM", "cs.DC", "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.AR", "cs.HC", "cs.IR", "cs.IT", "cs.LO", "cs.LG", "cs.MS", "cs.MA", "cs.MM", "cs.NI", "cs.NE", "cs.NA", "cs.OS", "cs.OH", "cs.PF", "cs.PL", "cs.RO", "cs.SI", "cs.SE", "cs.SD", "cs.SC", "cs.SY"],
    "Physics": ["physics.acc-ph", "physics.app-ph", "physics.ao-ph", "physics.atom-ph", "physics.atm-clus", "physics.bio-ph", "physics.chem-ph", "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-ph", "physics.med-ph", "physics.optics", "physics.ed-ph", "physics.soc-ph", "physics.plasm-ph", "physics.pop-ph", "physics.space-ph"],
    "Chemistry": ["nlin.AO", "nlin.CG", "nlin.CD", "nlin.SI", "nlin.PS"],
    "Statistics": ["stat.AP", "stat.CO", "stat.ML", "stat.ME", "stat.OT", "stat.TH"],
    "Biology": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"]
}

# Assign domain to each paper
def assign_domain(categories):
    for domain, domain_categories in domain_wise_categories.items():
        if any(cat in categories for cat in domain_categories):
            return domain
    return "Other"

df['domain'] = df['categories'].apply(assign_domain)

print("\nChecking label assignment:")
for domain, categories in domain_wise_categories.items():
    sample = df[df['domain'] == domain].sample(1)
    print(f"\nDomain: {domain}")
    print(f"Categories: {sample['categories'].values[0]}")
    print(f"Abstract: {sample['preprocessed_abstract'].values[0][:100]}...") 

# Perform clustering for each domain
domain_clusters = {}
for domain in domain_wise_categories.keys():
    domain_abstracts = df[df['domain'] == domain]['preprocessed_abstract'].tolist()
    if domain_abstracts:
        vectorizer = TfidfVectorizer()
        abstract_vectors = vectorizer.fit_transform(domain_abstracts)

        num_clusters = len(domain_wise_categories[domain])
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(abstract_vectors)

        domain_clusters[domain] = [domain_abstracts[i] for i in range(len(domain_abstracts))]

# Prepare data for fine-tuning
all_abstracts = [abstract for abstracts in domain_clusters.values() for abstract in abstracts]
all_labels = [domain for domain, abstracts in domain_clusters.items() for _ in abstracts]

# Prepare labels
le = LabelEncoder()
labels = le.fit_transform(all_labels)
num_labels = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(all_abstracts, labels, test_size=0.2, random_state=42, stratify=labels)

class ArXivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(X, y, tokenizer, batch_size=16):
    dataset = ArXivDataset(X, y, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def setup_model(num_labels, tokenizer):
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        model_type="bert",
        use_gradient_checkpointing=True,
        num_labels=num_labels
    )
    model = BertForSequenceClassification(config)
    try:
        pretrained_model = BertModel.from_pretrained(MODEL_PATH)
        model.bert = pretrained_model.bert
        print(f"Loaded pretrained weights from {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load pretrained weights from {MODEL_PATH}. Starting with random weights. Error: {e}")
    print(f"Initialized model with vocabulary size: {config.vocab_size}")
    
    return model

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=5e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = amp.GradScaler()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = torch.clamp(batch['input_ids'], max=model.config.vocab_size - 1).to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}:")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"attention_mask shape: {attention_mask.shape}")
                print(f"labels shape: {labels.shape}")
                print(f"input_ids max value: {input_ids.max().item()}")
                print(f"Vocab size: {model.config.vocab_size}")

            optimizer.zero_grad()

            try:
                with amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

            except RuntimeError as e:
                print(f"Error in forward/backward pass: {e}")
                continue

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = torch.clamp(batch['input_ids'], max=model.config.vocab_size - 1).to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                try:
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    val_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    val_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
                    val_samples += labels.size(0)
                except RuntimeError as e:
                    print(f"Error in validation: {e}")
                    continue

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        val_f1 = f1_score(val_true, val_preds, average='weighted')

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

    return model, train_losses, train_accuracies, val_losses, val_accuracies

def train_and_evaluate(tokenizer, name):
    print(f"\nTraining with {name} tokenizer:")
    print(f"Vocabulary size: {len(tokenizer)}")

    train_loader = create_data_loader(X_train, y_train, tokenizer)
    val_loader = create_data_loader(X_test, y_test, tokenizer)

    model = setup_model(num_labels, tokenizer)
    trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader)

    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{name} Tokenizer - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title(f'{name} Tokenizer - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'/gpfswork/rech/fmr/uft12cr/finetuneAli/results/{name}_tokenizer_plot.png')
    plt.close()

    # Evaluate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = torch.clamp(batch['input_ids'], max=model.config.vocab_size - 1).to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            test_preds.extend(outputs.logits.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)

    # One-hot encode the true labels
    # One-hot encode the true labels
    test_true_one_hot = np.eye(num_labels)[test_true]

    test_accuracy = accuracy_score(test_true, test_preds.argmax(axis=1))
    test_f1 = f1_score(test_true, test_preds.argmax(axis=1), average='weighted')
    test_auc = roc_auc_score(test_true_one_hot, test_preds, multi_class='ovr')

    print(f"\nTest Results for {name} tokenizer:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")

    return test_accuracy, test_f1, test_auc

# Load the general BERT tokenizer
try:
    all_cluster_tokenizer = BertTokenizer.from_pretrained('/gpfswork/rech/fmr/uft12cr/finetuneAli/ALL_clusters_vocab2.txt', local_files_only=True)
    final_tokenizer = BertTokenizer.from_pretrained('/gpfswork/rech/fmr/uft12cr/finetuneAli/final_vocab_cluster_pruningNew.txt', local_files_only=True)
    general_tokenizer = BertTokenizer.from_pretrained(General_TOKENIZER_PATH, do_lower_case=True, local_files_only=True)
except OSError as e:
    print(f"Failed to load one or more tokenizers. Error: {e}")
    raise

tokenizers = [all_cluster_tokenizer, final_tokenizer, general_tokenizer]
tokenizer_names = ['All Cluster', 'Final', 'General']

# Make sure the output directory exists
os.makedirs('/gpfswork/rech/fmr/uft12cr/finetuneAli/results', exist_ok=True)

# Train and evaluate with each tokenizer
results = {}
for tokenizer, name in zip(tokenizers, tokenizer_names):
    accuracy, f1, auc = train_and_evaluate(tokenizer, name)
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'AUC-ROC': auc}

# Print summary of results
print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name} Tokenizer:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Print class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution in training set:")
for u, c in zip(unique, counts):
    print(f"Class {le.inverse_transform([u])[0]}: {c} samples")
