import pandas as pd
import tensorflow as tf

# Load and preprocess image
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image

def create_dataset(df, batch_size=32):
    file_paths = df['filepaths'].values
    labels = df['class_index'].values

    if len(file_paths) == 0 or len(labels) == 0:
        raise ValueError("File paths or labels are empty.")

    file_paths_tensor = tf.constant(file_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)  # Ensure labels are int32

    def load_image_and_label(file_path, label):
        image = load_and_preprocess_image(file_path)
        return image, label

    def tf_load_image_and_label(file_path, label):
        image, label = tf.py_function(func=load_image_and_label,
                                      inp=[file_path, label],
                                      Tout=[tf.float32, tf.int32])
        image.set_shape([128, 128, 3])
        label.set_shape([])  # Ensure the label shape is scalar
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths_tensor, labels_tensor))
    dataset = dataset.map(tf_load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Load CSV data
df = pd.read_csv('beandata.csv')

# Split the data into training and validation sets
train_df = df[df['dataset'] == 'train']
val_df = df[df['dataset'] == 'test']

# Create the datasets
train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)

# Build the model
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),  # Explicitly define input shape
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Using sparse_categorical_crossentropy for integer labels
                  metrics=['accuracy'])
    return model

# Define input shape and number of classes
input_shape = (128, 128, 3)
num_classes = 4

# Print the shapes to ensure correctness
for images, labels in train_ds.take(1):
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

# Build and summarize the model
model = build_model(input_shape, num_classes)
model.summary()

# Train the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_ds)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('my_image_model.h5')

