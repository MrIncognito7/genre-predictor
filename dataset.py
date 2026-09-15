import numpy as np
import os
import librosa as lb
def audio_spec(chunk,sr=22050,n_mels=128):
    mel_spectrogram = lb.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def data_load(path,chunk_duration=3.0,sr=22050,n_mels=128):
    x,y = [],[]
    data = os.listdir(path)
    for genre in data:
        genre_path = os.path.join(path,genre)
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path,file)
            try:
                signal, _ = lb.load(file_path, sr=sr)
                samples_per_chunk = int(chunk_duration * sr)
                num_chunks = len(signal) // samples_per_chunk
                for i in range(num_chunks):
                    chunk = signal[i*samples_per_chunk:(i+1)*samples_per_chunk]
                    mel_spec = audio_spec(chunk, sr=sr, n_mels=n_mels)
                    x.append(mel_spec)
                    y.append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}") 
    return np.array(x), np.array(y)



X,Y = data_load('genres_original',chunk_duration=3.0,sr=22050,n_mels=128)
np.save('X_features.npy', X)
np.save('Y_labels.npy', Y)
print(X.shape)




