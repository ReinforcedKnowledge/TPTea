# TPTea
Tweaks and Pre-Training Tricks for Transformer-based Language Models.

## Description
This project is starting as a learning project for us to learn how each component of transformer based language models work in their utmost details and also discover the best practices and why they're the best practices.

## Roadmap
Set up the project as a package
- For config management and handling we can use/get inspired from Hydra and YACS, or other tools.
- Best practice to generate a Python package (setup, toml etc.)

Everything implemented must be tested (pytest), check out tests coverage
Check how to automate new tests, styling (isort, black/ruff), static type checks (mypy)

Implement the following components (the logic may be revisited)
- Dataset, check how we can make it distributed, 
- Model,
- Tokenizer (If needed, implement a faster version, either through algorithmic optimization or using another language),
- Optimizer,
- Attention,
- Embeddings,
- Activations,
- Normalization,
- Feed-Forward,
- Logger,
- Experiment Tracking,
- Trainer(Dataset, Model, Optimizer) -> runs optimization, maybe check distributed training
- Decoding Strategies (greedy, beam search etc.),
