import librosa
import pickle
import numpy as np

from keras.models import model_from_json

import warnings
warnings.filterwarnings('ignore')


def get_features(file):

    y, sr = librosa.load(file, mono=True)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = [np.mean(x) for x in librosa.feature.mfcc(y=y, sr=sr)]

    data = [chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr]
    data.extend(mfcc)

    return data


def load_model():

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    return loaded_model


def prediction(features):

    label_dict = pickle.load(open('target_dict', 'rb'))
    scaler = pickle.load(open('scaler.pickle', 'rb'))

    model = load_model()
    scaled_features = scaler.transform(np.reshape(features, (1, -1)))
    pred = np.argmax(model.predict(np.reshape(scaled_features, (1, 26))))

    return list(label_dict)[pred]