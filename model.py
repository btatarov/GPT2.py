import time
from typing import Optional

import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel

from dataset import Dataset
from log import Log


class Model:
    CHECKPOINT_DIR = './.checkpoint/'

    def __init__(self, dataset: Dataset, log: Log, rebuild: Optional[bool] = False) -> None:
        tokenizer = dataset.get_tokenizer()
        if rebuild:
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            self._model = TFGPT2LMHeadModel(config)
        else:
            self._model = TFGPT2LMHeadModel.from_pretrained(self.CHECKPOINT_DIR)
            self._model.load_weights(self.CHECKPOINT_DIR)

        self._log = log
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy('acc'),
            tf.keras.metrics.SparseCategoricalCrossentropy('ent'),
        ]

    def train(self, epochs: Optional[int] = 2) -> None:
        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        _ = self._model.fit(
            x=self._dataset['train'], validation_data=self._dataset['test'],
            initial_epoch=len(self._log), epochs=len(self._log) + epochs,
            callbacks=[
                ModelLogCallback(self),
            ],
        )

    def generate_text(self, seed_text: str, length: Optional[int] = 256) -> None:
        input_ids = self._tokenizer.encode(seed_text, return_tensors='tf')
        outputs = self._model.generate(
            input_ids,
            attention_mask=[[1] * len(input_ids[0])],
            pad_token_id=self._tokenizer.pad_token_id,
            max_length=length,
            num_beams=16,
            temperature=0.6,
            no_repeat_ngram_size=2,
            num_return_sequences=16,
        )
        print(self._tokenizer.decode(outputs[0], skip_special_tokens=True))

    def save_checkpoint(self) -> None:
        self._model.save_pretrained(self.CHECKPOINT_DIR)
        self._model.save_weights(self.CHECKPOINT_DIR)
        self._model.config.save_pretrained(self.CHECKPOINT_DIR)
        self._tokenizer.save_pretrained(self.CHECKPOINT_DIR)


class ModelLogCallback(tf.keras.callbacks.Callback):

    def __init__(self, model: Model) -> None:
        super().__init__()
        self._model = model

    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, float]] = dict()) -> None:
        self._time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, float]] = dict()) -> None:
        epoch_time = round(time.perf_counter() - self._time)
        self._model.save_checkpoint()
        self._model._log.add_epoch(epoch_time, logs)
