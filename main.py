from dataset import audio_spec
import os
import numpy as np
import joblib
import tensorflow as tf
import librosa as lb


def predict_genre(file_path, model, label_encoder, sr=22050, chunk_duration=3.0, n_mels=128):
    signal, _ = lb.load(file_path, sr=sr)

    samples_per_chunk = int(chunk_duration * sr)
    num_chunks = len(signal) // samples_per_chunk

    if num_chunks == 0:
        print("Audio file is too short needs at least 3 seconds.")
        return None
    chunks = []
    for i in range(num_chunks):
        chunk = signal[i*samples_per_chunk:(i+1)*samples_per_chunk]
        mel_spec = audio_spec(chunk, sr=sr, n_mels=n_mels)
        chunks.append(mel_spec)

    X_new = np.array(chunks)
    X_new = X_new[..., np.newaxis]                       
    X_new = (X_new - X_new.mean()) / X_new.std()          
    predictions = model.predict(X_new, verbose=0)         
    avg_prediction = np.mean(predictions, axis=0)

    genre_idx = np.argmax(avg_prediction)
    genre_name = label_encoder.inverse_transform([genre_idx])[0]
    confidence = avg_prediction[genre_idx] * 100

    # Show the top 3 guesses
    top3_idx = np.argsort(avg_prediction)[::-1][:3]
    print(f"\nPredicted genre: {genre_name.upper()} ({confidence:.1f}% confidence)")
    print("\nTop 3 predictions:")
    for idx in top3_idx:
        name = label_encoder.inverse_transform([idx])[0]
        print(f"  {name:12s} {avg_prediction[idx]*100:5.1f}%")

    return genre_name
if __name__ == "__main__":
    model = tf.keras.models.load_model('genre_model.keras')
    label_encoder = joblib.load('label_encoder.pkl')

    audio_file_path = input("Enter the path to the audio file: ")
    if os.path.isfile(audio_file_path):
        predict_genre(audio_file_path, model, label_encoder)
    else:
        print("Invalid file path. Please provide a valid audio file.")

