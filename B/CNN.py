import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam

def CNNRun():
    # Load the dataset
    data = np.load('./Datasets/pathmnist.npz')

    warnings.filterwarnings('ignore')

    x_train = data['train_images']
    y_train = data['train_labels'].ravel()
    x_val = data['val_images']
    y_val = data['val_labels'].ravel()
    x_test = data['test_images']
    y_test = data['test_labels'].ravel()

    # Normalize the data and use One-hot format to encode the labels
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Build the model architecture
    CNN_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), padding="same", input_shape=(28, 28, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")  # Adjusted to match the number of classes
    ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    CNN_model.compile(optimizer=Adam(learning_rate=0.001), 
                    loss=losses.categorical_crossentropy, # Select a loss function of cross-entropy that the network will minimize.
                    metrics=["accuracy"])

    # Train the model
    history = CNN_model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val), 
        batch_size=32,  # Adjusted batch size
        epochs=50,    # Increased epochs with EarlyStopping
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = CNN_model.evaluate(x_test, y_test)

    # Get predicted probabilities
    y_pred_test = CNN_model.predict(x_test)

    # Convert probabilities to label indices
    y_pred_labels = np.argmax(y_pred_test, axis=1)

    # Convert one-hot encoded y_test to label indices
    y_test_labels = np.argmax(y_test, axis=1)

    # Now use classification_report and print test loss, accuracy and confusion matrix
    test_report = classification_report(y_test_labels, y_pred_labels)
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    print(f"\n\nTest Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(test_report)
    
def Runall():
    CNNRun()
    
if __name__ == "__main__":
    Runall()

