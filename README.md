# CLAUSE: AGENTIC NEURO-SYMBOLIC KNOWLEDGE GRAPH REASONING VIA DYNAMIC LEARNABLE CONTEXT ENGINEERING

A reinforcement learning framework for multi-hop question answering over knowledge graphs, featuring multi-agent coordination and advanced graph traversal strategies.

## Features

- **Multi-Agent RL Framework**: Implements various RL algorithms (MAPPO, LC-MAPPO, IPPO, COPPO, GRPO)
- **Knowledge Graph Reasoning**: Enhanced graph traversal with FAISS-based similarity search
- **Question Answering**: End-to-end pipeline from question to answer generation
- **Multiple Datasets**: Support for MetaQA, FactQA, and HotpotQA datasets

## Project Structure

```
CLAUSE/
├── agents/                    # Agent implementations
│   ├── decoder.py            # Answer generation
│   ├── enhanced_graph_builder.py  # Graph construction
│   ├── reranker.py           # Answer reranking and API integration
│   └── traversal.py          # Graph traversal agents
├── rl_algo/                  # RL algorithm implementations
│   ├── mappo.py             # Multi-Agent PPO
│   ├── lc_mappo.py          # Loosely-Coupled MAPPO
│   ├── ippo.py              # Independent PPO
│   ├── coppo.py             # Coordinated PPO
│   ├── grpo.py              # Graph-based RPO
│   └── action_encoding.py   # Action space encoding
├── preprocessing/            # Data preprocessing
│   ├── extract_kg.py        # Extract knowledge graphs from datasets
│   └── subgraph_builder.py  # Build subgraph indexes
├── datasets/                 # Dataset directory (download separately)
├── cache/                    # Cache for embeddings and indexes
├── train.py                 # Training script
├── test.py                  # Testing/evaluation script
├── kg_env.py                # Knowledge graph environment
├── evaluator.py             # Evaluation metrics
├── metrics.py               # Performance metrics
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Erikdai/CLAUSE.git
   cd CLAUSE
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your SiliconFlow API key
   ```

## Dataset Setup

Download the required datasets from:

- **MetaQA**: https://github.com/yuyuz/MetaQA
- **FactQA**: https://github.com/aukhanee/FactQA
- **HotpotQA**: https://hotpotqa.github.io/

Place the downloaded datasets in the `datasets/` directory.

For FactQA and HotpotQA, preprocess the data first:
```bash
python preprocessing/extract_kg.py
```

## Usage

### 1. Build Graph Index Cache

Generate embeddings and FAISS indexes:
```bash
python preprocessing/subgraph_builder.py
```

### 2. Training

Train the model with your chosen RL algorithm:
```bash
python train.py --dataset metaqa --algo lc_mappo --epochs 10
```

### 3. Testing

Evaluate the trained model:
```bash
python test.py --dataset metaqa --checkpoint path/to/checkpoint.pt
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# SiliconFlow API Key (required for LLM-based components)
SILICONFLOW_API_KEY=your_api_key_here
```

### Command Line Arguments

Common training arguments:
- `--dataset`: Dataset name (metaqa, factqa, hotpotqa)
- `--algo`: RL algorithm (mappo, lc_mappo, ippo, coppo, grpo)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate

See `train.py` for full list of arguments.

## Algorithms

- **MAPPO**: Multi-Agent Proximal Policy Optimization
- **LC-MAPPO**: Loosely-Coupled MAPPO with reduced coordination overhead
- **IPPO**: Independent PPO for each agent
- **COPPO**: Coordinated PPO with explicit agent communication
- **GRPO**: Graph-aware Reinforcement Policy Optimization

## Requirements

- Python 3.8+
- PyTorch 1.11+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full dependencies

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

For questions or issues, please open an issue on GitHub or contact [daicxx1226@gmail.com](mailto:daicxx1226@gmail.com).

## Acknowledgments

This project builds upon research in multi-agent reinforcement learning and knowledge graph reasoning.
