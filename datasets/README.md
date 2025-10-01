# Dataset Directory

This directory contains the datasets used for knowledge graph reasoning and question answering.

## Dataset Links

Please download the required datasets from the following sources:

- **MetaQA**: https://github.com/yuyuz/MetaQA
- **FactQA**: https://github.com/aukhanee/FactQA
- **HotpotQA**: https://hotpotqa.github.io/

## Usage Notes

**Important**: FactQA and HotpotQA datasets require preprocessing before use with the CLAUSE system. Please use `extract_kg.py` from the `preprocessing` directory to extract questions and triples first.

```bash
python preprocessing/extract_kg.py
```
