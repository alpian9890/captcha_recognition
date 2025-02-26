import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import os
import random
import cv2
import matplotlib.pyplot as plt

def decode_prediction(pred, vocab):
    pred = pred.argmax(axis=1)
    decoded = "".join([vocab[c] for c in pred if c < len(vocab)])
    return decoded

def load_configs(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def inferno():
    # Path ke model dan data
    model_dir = "Models/captcha_to_text/202502201804/"
    model_path = os.path.join(model_dir, "model.h5")
    configs_path = os.path.join(model_dir, "configs.yaml")
    val_csv_path = os.path.join(model_dir, "val.csv")

    # Load konfigurasi
    configs = load_configs(configs_path)
    vocab = configs['vocab']

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load data validasi
    val_df = pd.read_csv(val_csv_path, header=None)
    val_df = val_df.iloc[1:] # Hapus baris header

    # Pilih gambar acak dari data validasi
    random_index = random.randint(0, len(val_df) - 1)
    image_path, true_label = val_df.iloc[random_index][0], val_df.iloc[random_index][1]

    # Load dan preprocess gambar
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (configs['width'], configs['height']))
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)

    # Prediksi
    prediction = model.predict(image)
    decoded_prediction = decode_prediction(prediction[0], vocab)

    # Tampilkan gambar dan hasil prediksi
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"True Label: {true_label}, Predicted: {decoded_prediction}")
    plt.show()

#if __name__ == "__main__":
    inferno()
