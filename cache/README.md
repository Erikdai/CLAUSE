# Cache Directory

This directory serves as a placeholder for automatically generated index files and caches.

## Notice

Due to size limitations, the complete index files are not included in the repository. The cache will be built automatically when you run the preprocessing scripts.

## Generated Contents

The following files will be automatically generated in this directory:

- Node embeddings and FAISS indexes
- Relation embeddings
- Graph structure caches
- Entity alignment mappings

## How to Generate Cache

1. Ensure your dataset is properly configured in the `datasets` directory
2. Run the subgraph builder:
   ```bash
   python preprocessing/subgraph_builder.py
   ```
3. The cache files will be automatically generated in this directory

**Note**: The initial cache generation may take some time depending on your dataset size.
