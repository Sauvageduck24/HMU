# Transformer Comparison Framework

This project provides a comprehensive framework for comparing standard Transformers with HTransformers in long-term memory and imagination tasks.

## Project Structure

```
.
├── requirements.txt
├── README.md
├── src/
│   ├── models/
│   │   ├── transformer.py
│   │   └── htransformer.py
│   ├── tasks/
│   │   ├── long_term_memory.py
│   │   └── imagination.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualization.py
└── tests/
    ├── test_long_term_memory.py
    └── test_imagination.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test suites:
```bash
pytest tests/test_long_term_memory.py
pytest tests/test_imagination.py
```

## Tasks

### Long-term Memory Tasks
- Sequential memory recall
- Context-dependent memory retrieval
- Memory persistence over extended sequences

### Imagination Tasks
- Future state prediction
- Counterfactual reasoning
- Creative sequence generation

## Evaluation Metrics

- Memory accuracy
- Sequence coherence
- Imagination quality
- Computational efficiency
- Memory persistence 