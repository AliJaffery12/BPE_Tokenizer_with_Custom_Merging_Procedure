# BPE_Tokenizer_with_Custom_Merging_Procedure
Apply BPE tokenizer with the Custom Merging Procedure for the domain specific dataset for Arxiv dataset.

1. Data Preprocessing (Preprocessing.py)
Text Cleaning: Tokenization, stopword removal, lemmatization, and vectorization (TF-IDF).
Clustering: Abstracts are clustered by domain using KMeans, and the results are saved to text files.

2. BPE Training (BPETraining.py)
BPE Implementation:
Custom class for BPE training with functions to calculate frequencies, apply merge rules, and train on batches.
Iterative merging of character pairs to create subword units.
Corpus Processing:
Each domain's abstracts are processed in batches, and BPE is trained on them.
The vocabulary for each cluster is extracted and saved.
Tokenization and Overlap Calculation:
Tokenizes texts using the trained BPE model.
Computes pairwise token overlap between different domain vocabularies and overall token overlap.
Vocabulary Merging and Pruning:
The final vocabulary is merged and refined by considering token coverage, frequency, and length.
The vocabulary is iteratively pruned by replacing underperforming tokens.
3. Evaluation Metrics
Subword Fertility: Measures how many subword units each word is divided into.
Text Compression: Evaluates the compression ratio of the text when tokenized by BPE.
Mean Token Length: Computes the average length of tokens in the vocabulary.
4. Saving Results
The final vocabulary after pruning is saved to a text file.

5. Final Metrics Calculation:

Complete the calculation of sf_final_vocab, tc_final_vocab, and mtl_final_vocab to evaluate the final BPE model effectively.
