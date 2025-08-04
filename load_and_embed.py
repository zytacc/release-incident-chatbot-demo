# load_and_embed.py

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load incident records
with open('release_incidents.json', 'r') as f:
    incidents = json.load(f)

# Combine relevant fields into a document string
documents = [
    f"{i['Date']} - {i['Issue Owner']}: {i['Symptom']} | Resolution: {i['Resolution']} | Root Cause: {i['Root Cause']} | Recommendation: {i['Recommended Solution']}"
    for i in incidents
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and documents for lookup
faiss.write_index(index, "incident_index.faiss")
with open("incident_texts.json", "w") as f:
    json.dump(documents, f, indent=2)

print("✅ Embeddings saved to FAISS index.")
