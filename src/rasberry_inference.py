import numpy as np
import librosa 
# import warnings
# warnings.filterwarnings("ignore")

def load_audio(pth):
	data, sr = librosa.load(pth, sr=16000)
	if data.size < 16000:
		data = np.pad(data, (16000-data.size, 0), mode='constant')

	return data, sr

def lfbe_delta(inp):
	mel_spec = librosa.feature.melspectrogram(inp, sr=16000, n_mels=13, hop_length=160, n_fft=480, fmin=20, fmax=4000)
	log_mel = librosa.core.power_to_db(mel_spec)
	lfbe_del = librosa.feature.delta(log_mel)
	lfbe_deldel = librosa.feature.delta(lfbe_del)
	features = np.vstack([log_mel,lfbe_del,lfbe_deldel])

	return np.array(features)

pth = 'sample.wav'
data, sr = load_audio(pth)
processed_audio = lfbe_delta(data)

print(processed_audio)  
