from concurrent.futures import process
import numpy as np
import librosa
import librosa.display
from tflite_runtime.interpreter import Interpreter
import argparse
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='rpi_inference')
parser.add_argument('--audio', action="store", dest="wavfile", required=True)
parser.add_argument(
    '--model',
    action="store",
    dest="tflitemodel",
    required=False)

args = parser.parse_args()
wavfile = args.wavfile
tflitemodel = args.tflitemodel


def load_audio(pth, plot=False):
    data, sr = librosa.load(pth, sr=16000)
    if data.size < 16000:
        data = np.pad(data, (16000 - data.size, 0), mode='constant')

    if plot:
        plt.plot(data)
        plt.show()

    return data, sr


def lfbe_delta(inp):
    mel_spec = librosa.feature.melspectrogram(
        inp,
        sr=16000,
        n_mels=13,
        hop_length=160,
        n_fft=480,
        fmin=20,
        fmax=4000)
    log_mel = librosa.core.power_to_db(mel_spec)
    lfbe_del = librosa.feature.delta(log_mel)
    lfbe_deldel = librosa.feature.delta(lfbe_del)
    features = np.vstack([log_mel, lfbe_del, lfbe_deldel])

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
    if word_scores[word_index] > 0:
        return word_dict[word_index]
    return "unknown"


def record_audio(record_duration):
    fs = 16000
    seconds = record_duration

    print(f'Listening...')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write('./data/recorded_audio.wav', fs, myrecording)
    print(f'Stopped listening')


def make_chunks(pth):
    sound_file = AudioSegment.from_wav(pth)
    audio_chunks = split_on_silence(sound_file, 
        # must be silent for at least half a second
        min_silence_len=100,

        # consider it silent if quieter than -16 dBFS
        silence_thresh=-25
    )

    chunk_pths = []
    for i, chunk in enumerate(audio_chunks):
        out_file = f"./data/chunk{i}.wav"
        chunk_pths.append(out_file)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")

    return chunk_pths

classes_map = {
    'follow': 0,
    'house': 1,
    'left': 2,
    'marvin': 3,
    'nine': 4,
    'no': 5,
    'right': 6,
    'seven': 7,
    'six': 8,
    'yes': 9
}
word_dict = {v: k for k, v in classes_map.items()}


def predict_audio(pth):
    data, _ = load_audio(pth)
    processed_audio = lfbe_delta(data)
    print(inference(args.tflitemodel, processed_audio, word_dict))


def predict_multiple_audio(pth):
    chunk_pths = make_chunks(pth)
    for chunk_pth in chunk_pths:
        predict_audio(chunk_pth)


def predict_single_audio(pth):
    predict_audio(pth)


def main():
    pth = args.wavfile
    if pth == 'record':
        record_audio(5)
        pth = './data/recorded_audio.wav'
    predict_multiple_audio(pth)


if __name__ == '__main__':
    main()

