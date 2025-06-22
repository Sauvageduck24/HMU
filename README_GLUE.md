### 6. MNLI (Multi-Genre Natural Language Inference)
- **Tarea**: Inferencia natural multi-género (3 clases)
- **Descripción**: Determinar si una premisa implica, contradice o es neutral respecto a una hipótesis
- **Tamaño**: ~392,702 ejemplos de entrenamiento, ~9,815 de validación (matched), ~9,832 de validación (mismatched)
- **Ejemplo**:
  - Premise: "The cat is on the mat"
  - Hypothesis: "There is a cat on the mat"
  - Label: 0 (entailment)

## Uso

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar Experimentos

#### Entrenamiento y Evaluación Completa
```bash
# MRPC
python run_experiments.py --dataset mrpc --mode all --epochs 5 --batch_size 16

# RTE
python run_experiments.py --dataset rte --mode all --epochs 5 --batch_size 16

# CoLA
python run_experiments.py --dataset cola --mode all --epochs 5 --batch_size 16

# QNLI
python run_experiments.py --dataset qnli --mode all --epochs 3 --batch_size 16

# SST-2
python run_experiments.py --dataset sst2 --mode all --epochs 5 --batch_size 16

# MNLI
python run_experiments.py --dataset mnli --mode all --epochs 3 --batch_size 16
``` 