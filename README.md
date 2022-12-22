# GPT2.py

A lightweight Tensorflow wrapper around HuggingFace's GPT2 language modeling head model that handles BPE tokenization, dataset preparation, training, and text generation. Epoch stats are saved to a JSON log file and the data is visualized using **matplotlib**.

## Requirements

Python 3.9+

## Usage

Use `main.py` as a starting point. Run the script with `--rebuild` the first time to generate the BPE tokens and datesets.

``` bash
./main.py [--rebuild]
```

## Datasets

Put the content you want to train the model on as UTF-8 encoded plain .txt files in a directory `./texts`.

### Constants

- `VOCAB_SIZE` - size of the BPE-tokenized vocabulary
- `DATASET_SEQ_LENGTH` - length of a single text chunk used for training
- `DATASET_BATCH_SIZE` - number of text chunks in a single training batch
- `DATASET_BUFFER_SIZE` - size of the buffer to use for dataset randomization
- `DATASET_TRAIN_DIR` - path to the directory where the generated training dataset is stored
- `DATASET_TEST_DIR` - path to the directory where the generated validation dataset is stored
- `TOKENIZED_DATA_DIR` - path to the directory where the generated vocabulary data is stored
- `CONTENT_DIR` - path to the directory where the raw data used for creating datasets is stored

## Model

The model is build on top of **TFGPT2LMHeadModel**. Can be used for training from scratch, or by loading a previous checkpoint.

### Constants

- `CHECKPOINT_DIR` - path to the directory where the latest checkpoint is stored

## Log

Saves epoch stats to a JSON log file. Currently, the following metrics are tracked: **loss**, **accuracy**, and **crossentropy**. Data is visualized with **matplotlib** using the following code:

``` python
from .log import Log

log = Log()
log.plot()
```

### Constants

- `LOG_FILE` - path to the JSON file used for logging
- `PLOT_DIR` - path to the directory where plot images are saved
