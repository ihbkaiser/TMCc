from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & good

# Read all documents (each line is one document)
with open("tm_datasets/NYT/train_texts.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# Generate embeddings
embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)

# Save embeddings to .npz with the key 'arr_0' to be compatible with your loading code
np.savez_compressed("tm_datasets/NYT/with_bert/train_bert.npz", embeddings)

print(f"Saved {len(embeddings)} embeddings to document_embeddings.npz")
