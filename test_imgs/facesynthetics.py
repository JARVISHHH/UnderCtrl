import tensorflow as tf
from datasets import load_dataset
import numpy as np

hf_dataset = load_dataset("multimodalart/facesyntheticsspigacaptioned", split='train')

def preprocess_example(example, img_size):
    target = example["image"].convert("RGB").resize((img_size, img_size))
    source = example["spiga_seg"].convert("RGB").resize((img_size, img_size))

    target = np.array(target).astype(np.float32)
    source = np.array(source).astype(np.float32)

    source = source / 255.0
    target = (target / 127.5) - 1.0

    return {
        "jpg": target,
        "hint": source,
        "txt": example["image_caption"]
    }

def generator(img_size):
    for example in hf_dataset:
        yield preprocess_example(example)


def get_dataset(batch_size=8, img_size=256):
    output_signature = {
        "jpg": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "hint": tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        "txt": tf.TensorSpec(shape=(), dtype=tf.string),
    }

    dataset = tf.data.Dataset.from_generator(
        lambda: generator(img_size),
        output_signature=output_signature
    )

    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Usage example
# dataset = get_dataset()