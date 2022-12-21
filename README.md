# GPT2.py

A lightweight Tensorflow wrapper around HuggingFace's GPT2 model that handles BPE tokenization, dataset preparation, training, and text generation.

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

- `VOCAB_SIZE` - size of the BPI-tokenized vocabulary
- `DATASET_SEQ_LENGTH` - length of a single text chunk used for training
- `DATASET_BATCH_SIZE` - number of text chunks in a single training batch
- `DATASET_BUFFER_SIZE` - size of the buffer to use for dataset randomization
- `DATASET_TRAIN_PATH` - path to the generated training dataset
- `DATASET_TEST_PATH` - path to the generated validation dataset
- `TOKENIZED_DATA_PATH` - path to the generated vocabulary data
- `CONTENT_PATH` - path to the raw data used for creating datasets

## Model

The model is build on top of **TFGPT2LMHeadModel**. Can be used for training from scratch, or by loading a previous checkpoint.

### Constants

- `CHECKPOINT_DIR` - path to the directory where the latest checkpoint is stored
