import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS_TL = 5         # Transfer learning phase
EPOCHS_FT = 5         # Fine-tuning phase

# 1️⃣ Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert grayscale (28x28) to RGB (96x96x3)
x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
x_test = np.repeat(x_test[..., np.newaxis], 3, -1)

# Resize to IMG_SIZE
x_train = tf.image.resize(x_train, IMG_SIZE) / 255.0
x_test = tf.image.resize(x_test, IMG_SIZE) / 255.0

# 2️⃣ Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# 3️⃣ Build MobileNetV2 transfer learning model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # Freeze base

model_tl = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model_tl.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4️⃣ Train transfer learning model
print("\n🔹 Training Transfer Learning Model...")
model_tl.fit(
    train_gen,
    validation_data=(x_test, y_test),
    epochs=EPOCHS_TL
)

# Save transfer learned model
model_tl.save("mnist_mobilenetv2_tl.keras")
print("✅ Saved mnist_mobilenetv2_tl.keras")

# 5️⃣ Fine-tuning: unfreeze some layers
base_model.trainable = True
fine_tune_at = 100  # Unfreeze from this layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model_tl.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔹 Fine-tuning Model...")
model_tl.fit(
    train_gen,
    validation_data=(x_test, y_test),
    epochs=EPOCHS_FT
)

# Save fine-tuned model
model_tl.save("mnist_mobilenetv2_finetuned.keras")
print("✅ Saved mnist_mobilenetv2_finetuned.keras")
