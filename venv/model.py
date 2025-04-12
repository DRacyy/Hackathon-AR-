import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def build_cnn_model(input_shape, num_classes=3):
    """
    Build and compile a CNN model for sequence prediction
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(), 
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # 3 classes: Sell, Hold, Buy
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def prepare_training_data(X, y, test_size=0.2):
    """
    Prepare data for training (convert labels and split dataset)
    """
    # Convert labels from -1/0/1 → 0/1/2
    y_adjusted = y + 1  # now: 0 = SELL, 1 = HOLD, 2 = BUY
    y_cat = to_categorical(y_adjusted, num_classes=3)
    
    # Train-test split (keeping time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X, y, epochs=20, batch_size=32, validation_split=0.1):
    """
    Train the model and return model, history, and evaluation results
    """
    X_train, X_test, y_train, y_test = prepare_training_data(X, y)
    
    model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split
    )
    
    # Evaluate the model
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    return model, history, X_test, y_test, y_pred_classes, y_true_classes

def plot_confusion_matrix(y_true_classes, y_pred_classes):
    """
    Plot confusion matrix of model predictions
    """
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Sell', 'Hold', 'Buy'], 
        yticklabels=['Sell', 'Hold', 'Buy']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_model(model, filepath='model.h5'):
    """
    Save the trained model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_trained_model(filepath='model.h5'):
    """
    Load a previously trained model
    """
    return load_model(filepath)

if __name__ == "__main__":
    # Test model training independently
    import asyncio
    from data_loader import fetch_data
    from feature_engineering import engineer_features, normalize_features, create_labels, prepare_sequences
    
    async def test_model():
        df = await fetch_data()
        df = engineer_features(df)
        df, features = normalize_features(df)
        df = create_labels(df)
        X, y = prepare_sequences(df, features)
        
        model, history, X_test, y_test, y_pred_classes, y_true_classes = train_model(X, y, epochs=5)
        plot_confusion_matrix(y_true_classes, y_pred_classes)
        plot_training_history(history)
        
    asyncio.run(test_model())