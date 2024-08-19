# BPE_Tokenizer_with_Custom_Merging_Procedure
Apply BPE tokenizer with the Custom Merging Procedure for the domain specific dataset for Arxiv dataset.

# 1. Data Preprocessing (Preprocessing.ipynb)
   
# Text Cleaning: 
Tokenization, stopword removal, lemmatization, and vectorization (TF-IDF).
# Clustering: 
Abstracts are clustered by domain using KMeans, and the results are saved to text files.

# 2. BPE Training (BPE_Training_with_Merging_Procedure.ipynb)
# BPE Implementation:
Custom class for BPE training with functions to calculate frequencies, apply merge rules, and train on batches.
Iterative merging of character pairs to create subword units.

# Corpus Processing:
Each domain's abstracts are processed in batches, and BPE is trained on them.
The vocabulary for each cluster is extracted and saved.

# Tokenization and Overlap Calculation:
Tokenizes texts using the trained BPE model.
Computes pairwise token overlap between different domain vocabularies and overall token overlap.

# Vocabulary Merging and Pruning:
The final vocabulary is merged and refined by considering token coverage, frequency, and length.
The vocabulary is iteratively pruned by replacing underperforming tokens.
# 3. Evaluation Metrics
# Subword Fertility: 
Measures how many subword units each word is divided into.
# Text Compression: 
Evaluates the compression ratio of the text when tokenized by BPE.
# Mean Token Length: 
Computes the average length of tokens in the vocabulary.
# 4. Saving Results
The final vocabulary after pruning is saved to a text file.

# 5. Final Metrics Calculation:

Complete the calculation of sf_final_vocab, tc_final_vocab, and mtl_final_vocab to evaluate the final BPE model effectively.

# 6. Fine-Tuning BERT for ArXiv Domain Classification

# Model Overview
# BERT (Bidirectional Encoder Representations from Transformers): 
A state-of-the-art model for various NLP tasks, including text classification.
# BERT Fine-Tuning: 
This involves starting with a pre-trained BERT model and training it further on a specific dataset (in this case, the ArXiv abstracts) to adapt it to a particular task.
# Preprocessing
# Text Cleaning:
Abstracts are converted to lowercase, and special characters are removed to standardize the text.
# Domain Assignment: 
Papers are assigned to a domain based on their ArXiv subject categories.
# Clustering: 
Within each domain, abstracts are clustered to help the model learn domain-specific features.
# Model Training
# Custom Dataset Class: 
A PyTorch Dataset class is used to handle text data and labels, ensuring compatibility with BERT's input format.
# DataLoader: 
The data is split into training and validation sets, and DataLoaders are created to feed the data into the model during training.
# Model Setup: 
A BERT model is configured with the appropriate number of output labels corresponding to the number of domains.
# Training Loop:
The model is trained for a specified number of epochs, with the optimizer, learning rate, and other hyperparameters configured. The training loop includes:
Forward pass
Loss computation
Backward pass and optimization
Gradient scaling using AMP (Automatic Mixed Precision) for faster training on GPUs.
# Tokenizers
Three different tokenizers are used to train and evaluate the model:

# All Cluster Tokenizer
# Final Tokenizer
# General Tokenizer
These tokenizers have different vocabularies and are expected to influence the model's performance differently.

# Evaluation
The model's performance is evaluated using the following metrics:

# Accuracy:
The proportion of correctly classified papers.
# F1 Score:
The weighted average of precision and recall, especially useful in imbalanced datasets.
# AUC-ROC: 
The Area Under the Receiver Operating Characteristic curve, which evaluates the model's ability to distinguish between classes.
# Results
The training process outputs:

Loss and accuracy plots for both training and validation sets.
Final performance metrics (Accuracy, F1 Score, AUC-ROC) for each tokenizer.
A summary of results comparing the performance across different tokenizers.


# How to Run
# Set Up Environment:

Ensure PyTorch, Transformers, and other dependencies are installed.
Make sure the BERT model and tokenizers are available locally.
# Execute the Script:

The script can be executed to start the fine-tuning process.
Results will be saved in the specified output directory.
# View Results:

Check the loss and accuracy plots in the results directory.
Review the printed summary of performance metrics for each tokenizer.
# Class Distribution:
The script also prints the distribution of classes in the training set to ensure a balanced dataset, which is crucial for training a robust model.

# Requirements
Python 3.x,
PyTorch,
Transformers library from Hugging Face,
Scikit-learn,
Matplotlib,
Pandas,
NumPy
