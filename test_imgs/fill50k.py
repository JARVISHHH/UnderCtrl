import json
import cv2
import numpy as np
import tensorflow as tf
from datasets import Dataset  # Hugging Face datasets

def read_prompt_dataset(path='./training/fill50k/prompt.json'):
    with open(path, 'rt') as f:
        items = [json.loads(line) for line in f]
    return Dataset.from_list(items)

def preprocess_item(item, img_size, model):
    source_path = './training/fill50k/' + item['source']
    target_path = './training/fill50k/' + item['target']
    prompt = item['prompt']

    encoded_text = model.encode_text(prompt)
    encoded_text = np.squeeze(encoded_text, axis=0)

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    source = cv2.resize(source, (img_size, img_size)).astype(np.float32) / 255.0
    target = cv2.resize(target, (img_size, img_size)).astype(np.float32)
    target = (target / 127.5) - 1.0

    return {
        'jpg': target,
        'hint': source,
        'txt': encoded_text,
        'str': prompt
    }

def make_generator(dataset, img_size, model):
    def gen():
        for item in dataset:
            yield preprocess_item(item, img_size, model)
    return gen

def get_dataset(model, batch_size, img_size, shuffle_seed=42, test_size=0.2):
    prompt_dataset = read_prompt_dataset()
    prompt_split = prompt_dataset.train_test_split(test_size=test_size, seed=shuffle_seed)

    train_items = prompt_split["train"]
    test_items = prompt_split["test"]

    output_signature = {
        'jpg': tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        'hint': tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        'txt': tf.TensorSpec(shape=(77, 768), dtype=tf.float32),
        'str': tf.TensorSpec(shape=(), dtype=tf.string)
    }

    train_dataset = tf.data.Dataset.from_generator(
        make_generator(train_items, img_size, model),
        output_signature=output_signature
    )

    test_dataset = tf.data.Dataset.from_generator(
        make_generator(test_items, img_size, model),
        output_signature=output_signature
    )

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, len(prompt_dataset)
