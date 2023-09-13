import tensorflow as tf
from model import get_model
from preprocess_data import get_data
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def train_model(start_index, end_index):
    """
    Trains the model on the data from the specified indices.

    Args:
        start_index (int): The start index of the data to use.
        end_index (int): The end index of the data to use.
    """

    x, y = get_data(start_index, end_index)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.05, random_state=42
    )
    x.shape, x_val.shape

    # Create the model
    model = get_model()

    # Compile the model
    optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    # Define callbacks
    checkpoint = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train the model with callbacks
    history = model.fit(
        x_train,
        y_train,
        shuffle=True,
        batch_size=64,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, reduce_lr, early_stopping],
    )

    # Load the best model weights
    model.load_weights("best_model.h5")

    # Save the best model in SavedModel
    tf.saved_model.save(model, "saved_model")

