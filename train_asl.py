import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, "data")
IMG_SIZE = (96, 96)

# 1. Chargement avec Data Augmentation intégrée
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_path, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_path, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=32
)

# 2. Modèle Transfer Learning amélioré
base_model = applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    data_augmentation, # Appliqué uniquement pendant l'entraînement
    layers.Rescaling(1./127.5, offset=-1), # Normalisation incluse ici !
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Évite le sur-apprentissage (overfitting)
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Entraînement avec plus d'époques
# On passe à 10 ou 15 époques pour laisser le temps à la Data Augmentation d'agir
model.fit(train_ds, validation_data=val_ds, epochs=15)

model.save("asl_model.h5")
print(f"✅ Modèle optimisé créé avec {len(train_ds.class_names)} classes.")