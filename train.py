import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
X = np.load('X_features.npy')
Y = np.load('Y_labels.npy')
import joblib

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D((128),(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),   
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, Y, test_size=0.2, random_state=42, epochs=50, batch_size=32):
    # Encode labels
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    X = X[..., np.newaxis]
    X = (X-X.mean())/X.std()

    
    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=test_size, random_state=random_state,stratify=Y_encoded)

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(label_encoder.classes_))
    history = model.fit(X_train, Y_train,validation_split=0.1,epochs=epochs, batch_size=batch_size,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)], verbose=1)
    return model, history, label_encoder,X_test, Y_test

acc,history,label_encoder,X_test, Y_test= train_model(X, Y, test_size=0.2, random_state=42, epochs=50, batch_size=32)
test_loss, test_accuracy = acc.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
acc.save('genre_model.keras')
joblib.dump(label_encoder, 'label_encoder.pkl')