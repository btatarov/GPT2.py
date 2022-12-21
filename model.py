from typing import Optional

import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel

from dataset import Dataset


# TODO: temporary
class StopTrainingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, float]] = None) -> None:
        self.model.stop_training = True


class Model:
    CHECKPOINT_DIR = './checkpoint/'

    def __init__(self, dataset: Dataset, rebuild: Optional[bool] = False) -> object:
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

        self._dataset = dataset
        self._tokenizer = tokenizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    def train(self, epochs: Optional[int] = 2) -> None:
        self._model.compile(
            optimizer=self._optimizer,
            loss=[self._loss, *[None] * self._model.config.n_layer],
            metrics=[self._metric,],
        )

        try:
            _ = self._model.fit(
                x=self._dataset['train'], validation_data=self._dataset['test'], epochs=epochs,
                callbacks=[
                    StopTrainingCallback(),
                ]
            )
        finally:
            self.save_checkpoint()

    def generate_text(self, seed_text: str, length: Optional[int] = 256) -> None:
        input_ids = self._tokenizer.encode(seed_text, return_tensors='tf')
        outputs = self._model.generate(
            input_ids,
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
