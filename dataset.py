import os
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer


class Dataset:
    VOCAB_SIZE = 100000
    DATASET_SEQ_LENGTH = 100
    DATASET_BATCH_SIZE = 35
    DATASET_BUFFER_SIZE = 5000
    DATASET_TRAIN_PATH = './datasets/train/'
    DATASET_TEST_PATH = './datasets/test/'
    TOKENIZED_DATA_PATH = './datasets/tokens/'
    CONTENT_PATH = './text/'

    def __init__(self, dataset_split_ratio: Optional[float] = 0.95, rebuild: Optional[bool] = False) -> None:
        if 1 <= dataset_split_ratio <= 0:
            raise ValueError('dataset_split_ratio must be in the range of (0, 1)')

        self._split_ratio = dataset_split_ratio
        self._tokenizer = None
        self._datasets = dict()

        if rebuild:
            self.generate_data(tokenize=True)
        else:
            self.load_data()

    def __getitem__(self, index: str) -> tf.data.Dataset:
        if not isinstance(index, str) or not index in ['train', 'test']:
            raise TypeError('Not a valid index. Valid options are "train" or "test"')

        return self._datasets[index]

    def get_tokenizer(self) -> GPT2Tokenizer:
        if not self._tokenizer:
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.TOKENIZED_DATA_PATH)
            self._tokenizer.add_special_tokens({
                'eos_token': '</s>',
                'bos_token': '<s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>',
                'mask_token': '<mask>',
            })
        return self._tokenizer

    def generate_data(self, tokenize: Optional[bool] = False) -> None:
        corpus_paths = sorted([str(path) for path in Path(self.CONTENT_PATH).glob('**/*.txt')])

        # BPE tokenize
        if tokenize:
            if not os.path.exists(self.TOKENIZED_DATA_PATH):
                os.makedirs(self.TOKENIZED_DATA_PATH)
            bpe_tokenizer = ByteLevelBPETokenizer(lowercase=True, unicode_normalizer='nfkc')
            bpe_tokenizer.add_tokens(['…',])
            bpe_tokenizer.train(
                files=corpus_paths,
                vocab_size=self.VOCAB_SIZE,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'],
            )
            bpe_tokenizer.save_model(self.TOKENIZED_DATA_PATH)

        # tokenize the raw texts
        tokenizer = self.get_tokenizer()
        all_texts = list()
        for path in corpus_paths:
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        tokenized_text = tokenizer.encode(tokenizer.eos_token.join(all_texts))

        # prepare datasets
        samples = [
            tokenized_text[i:i + self.DATASET_SEQ_LENGTH]
            for i in range(0, len(tokenized_text) - self.DATASET_SEQ_LENGTH + 1, self.DATASET_SEQ_LENGTH)
        ]
        data = [(sample[:-1], sample[1:]) for sample in samples if not tokenizer.unk_token_id in sample]
        inputs, labels = tuple(map(list, zip(*data)))

        batches = round(len(inputs) * self._split_ratio)
        dataset_train = tf.data.Dataset.from_tensor_slices((inputs[:batches], labels[:batches]))
        tf.data.Dataset.save(dataset_train, self.DATASET_TRAIN_PATH, compression='GZIP')

        dataset_test = tf.data.Dataset.from_tensor_slices((inputs[batches:], labels[batches:]))
        tf.data.Dataset.save(dataset_test, self.DATASET_TEST_PATH, compression='GZIP')

        self.load_data(dataset_train, dataset_test)

    def load_data(self, dataset_train: Optional[tf.data.Dataset] = None, dataset_test: Optional[tf.data.Dataset] = None) -> None:
        dataset_train = dataset_train or tf.data.Dataset.load(self.DATASET_TRAIN_PATH, compression='GZIP')
        dataset_train = dataset_train.shuffle(self.DATASET_BUFFER_SIZE)
        dataset_train = dataset_train.batch(self.DATASET_BATCH_SIZE, drop_remainder=True)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)
        self._datasets['train'] = dataset_train

        dataset_test = dataset_test or tf.data.Dataset.load(self.DATASET_TEST_PATH, compression='GZIP')
        dataset_test = dataset_test.shuffle(self.DATASET_BUFFER_SIZE)
        dataset_test = dataset_test.batch(self.DATASET_BATCH_SIZE, drop_remainder=True)
        dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)
        self._datasets['test'] = dataset_test
