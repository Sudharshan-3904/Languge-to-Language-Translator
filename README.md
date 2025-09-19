# L2LTranslator

L2LTranslator is a bilingual sequence-to-sequence translation project. It provides tools and scripts for training, evaluating, and using neural translation models on paired bilingual data. The project supports both Keras and PyTorch models and is designed for extensibility to new language pairs.

## Directory Structure

```
main.py                  # Main entry point for running translation tasks
requirements.txt         # Python dependencies
train_notebook.ipynb     # Jupyter notebook for interactive training and analysis
train_on_bilingual.py    # Script for training on bilingual data
train.py                 # General training script
translator.py            # Translation model and utilities
bilingual_pairs/         # Contains bilingual text files (e.g., fra.txt, tam.txt)
savepoints/              # Model checkpoints and vocabulary files
    lengths.pkl
    model_savepoint.keras
    model_seq2seq.pt
    src_vocab.pkl
    tgt_vocab.pkl
```

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Place your bilingual text files in `bilingual_pairs/`.
   - Format: Each line in `src` and `tgt` files should correspond.

3. **Train a Model**
   - Use `train.py` or `train_on_bilingual.py` for command-line training.
   - Use `train_notebook.ipynb` for interactive experimentation.

4. **Checkpoints & Vocabularies**
   - Model checkpoints and vocabulary files are saved in `savepoints/`.
   - Supports both Keras (`.keras`) and PyTorch (`.pt`) formats.

5. **Translate**
   - Use `main.py` or `translator.py` to perform translation using trained models.

## Scripts Overview

- **main.py**: Main script for translation tasks.
- **train.py**: General training script, may support multiple configurations.
- **train_on_bilingual.py**: Specialized training on paired bilingual data.
- **translator.py**: Contains translation logic and model utilities.
- **train_notebook.ipynb**: Interactive notebook for training and analysis.

## Data Format

- `bilingual_pairs/fra.txt`: Source language sentences (e.g., French)
- `bilingual_pairs/tam.txt`: Target language sentences (e.g., Tamil)
- Each line in both files should be aligned.

## Model Savepoints

- `savepoints/model_savepoint.keras`: Keras model checkpoint
- `savepoints/model_seq2seq.pt`: PyTorch model checkpoint
- `savepoints/src_vocab.pkl`, `savepoints/tgt_vocab.pkl`: Pickled vocabularies
- `savepoints/lengths.pkl`: Sequence length metadata

## Requirements

See `requirements.txt` for all dependencies. Typical requirements include TensorFlow, PyTorch, NumPy, etc.

## Usage Example

```bash
python train_on_bilingual.py --src bilingual_pairs/fra.txt --tgt bilingual_pairs/tam.txt
python main.py --input "Bonjour" --model savepoints/model_savepoint.keras
```

## Extending to New Languages

- Add new bilingual text files to `bilingual_pairs/`.
- Update scripts to point to new data files.

## License

Specify your license here (e.g., MIT, Apache 2.0).

## Contact

For questions or contributions, open an issue or contact the maintainer.
