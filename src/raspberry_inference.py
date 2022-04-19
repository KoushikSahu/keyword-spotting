from concurrent.futures import process
import numpy as np
import librosa 
from tflite_runtime.interpreter import Interpreter 
import argparse

parser = argparse.ArgumentParser(description='rpi_inference')
parser.add_argument('--audio',action="store",dest="wavfile",required=False)
parser.add_argument('--model',action="store",dest="tflitemodel",required=False)

argss = parser.parse_args()
wavfile = argss.wavfile
tflitemodel = argss.tflitemodel
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

def inference(model_path, processed_audio, word_dict):
	interpreter = Interpreter(model_path=model_path)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Make prediction from model
	in_tensor = np.float32(processed_audio)
	in_batch_tensor = []
	for _ in range(16):
		in_batch_tensor.append(in_tensor)
	in_batch_tensor = np.float32(in_batch_tensor)

	interpreter.set_tensor(input_details[0]['index'], in_batch_tensor)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	word_scores = output_data[0]
	print(word_scores)
	word_index = np.argmax(word_scores, axis=0)
	if word_scores[word_index]>2:
		return word_dict[word_index]
	return "unknown"

def main():
	pth = argss.wavfile
	data, _ = load_audio(pth)
	processed_audio = lfbe_delta(data)
	classes_map ={'follow': 0, 'house': 1, 'left': 2, 'marvin': 3, 'nine': 4, 'no': 5, 'right': 6, 'seven': 7, 'six': 8, 'yes': 9}
	word_dict = {v: k for k, v in classes_map.items()}
	print(inference(argss.tflitemodel, processed_audio, word_dict))

if __name__ == '__main__':
	main()
