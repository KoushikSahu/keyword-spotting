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
from timeit import default_timer as timer
from scipy.special import softmax

parser = argparse.ArgumentParser(description='rpi_inference')
parser.add_argument('--audio', action = "store", dest = "wavfile", required = True)
parser.add_argument(
    '--model',
    action = "store",
    dest = "tflitemodel",
    required = True)

args = parser.parse_args()
wavfile = args.wavfile
tflitemodel = args.tflitemodel


class HelperClass:
    classes_map = {
        'off': 0,
        'on': 1,
        'one': 2,
        'zero': 3
    }
    word_dict = {v: k for k, v in classes_map.items()}

    @staticmethod
    def lfbe_delta(inp):
        mel_spec = librosa.feature.melspectrogram(
            y=inp,
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


class Inference:
    def __init__(self):
        self.interpreter = Interpreter(model_path=tflitemodel)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def load_audio(self, pth, plot=False):
        data, sr = librosa.load(pth, sr=16000)
        if data.size < 16000:
            data = np.pad(data, (16000 - data.size, 0), mode='constant')

        if plot:
            plt.plot(data)
            plt.xlabel('Sampling Rate X Number of seconds')
            plt.ylabel('Amplitude')
            plt.show()

        return data, sr


    def record_audio(self,
            record_duration,
            sampling_rate=16000,
            save_pth='./data/recorded_audio.wav'):
        fs = sampling_rate
        seconds = record_duration

        print(f'Listening...')
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_pth, fs, myrecording)
        print(f'Stopped listening')


    def inference(self, processed_audio, word_dict, threshold=0.5, timeit=True):
        if timeit:
            start = timer()

        in_tensor = np.float32(processed_audio)
        in_tensor = in_tensor.reshape((1, 39, 101))
        self.interpreter.set_tensor(self.input_details[0]['index'], in_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        word_scores = output_data[0]
        word_softmax = softmax(word_scores, axis=0)
        print(f'word softmax: {word_softmax}')
        word_index = np.argmax(word_softmax, axis=0)
        print(f'word probability: {word_softmax[word_index]}')

        if timeit:
            end = timer()
            print(f'Inference time: {end - start}')

        if word_scores[word_index] >= threshold:
            return word_dict[word_index]
        return "unknown"


    def make_chunks(self, pth):
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


    def make_audio_chunks(self, rec, save_pth='./data/rec.wav'):
        fs = 16000
        pth = save_pth

        write(pth, fs, rec)
        chunk_pths = self.make_chunks(pth)
        return chunk_pths


    def predict_audio(self, pth, feature_extraction=HelperClass.lfbe_delta, class_map=HelperClass.word_dict, verbose=False):
        if verbose:
            start = timer()

        if pth == 'record':
            self.record(1)
        data, _ = self.load_audio(pth)
        load_end = timer()

        if verbose:
            print(f'Loading time: {load_end - start}')

        processed_audio = feature_extraction(data)

        if verbose:
            lfbe_end = timer()
            print(f'Preprocessing time: {lfbe_end - load_end}')

        predicted_word = self.inference(processed_audio, class_map)

        if verbose:
            end = timer()
            print(f'Inference time: {end - lfbe_end}')
            
        print(f'Predicted word: {predicted_word}')


    def predict_single_audio(self, pth):
        if pth == 'record':
            self.record_audio(1)
            pth = './data/recorded_audio.wav'
        self.predict_audio(pth)



    def chunking_audio_prediction(self, pth, timeit=False):
        if pth == 'record':
            self.record_audio(5)
            pth = './data/recorded_audio.wav'

        if timeit:
            start = timer()

        chunk_pths = self.make_chunks(pth)
        for chunk_pth in chunk_pths:
            self.predict_audio(chunk_pth)

        if timeit:
            end = timer()
            print(f'Total Inference Latency: {end - start}')


    def sliding_window(self, feature_extraction=HelperClass.lfbe_delta, class_map=HelperClass.word_dict):
        num_channels = 2
        sample_rate = 16000
        rec_duration = 0.5
        window = np.zeros(int(rec_duration*sample_rate)*2)

        print(f'Listening...')
        def sd_callback(rec, frames, time, status):
            rec = rec[:, 0]
            rec = np.squeeze(rec)
            window[:len(window)//2] = window[len(window)//2:]
            window[len(window)//2:] = rec
            processed_audio = feature_extraction(window)
            res = self.inference(processed_audio, class_map)
            print(res)
        with sd.InputStream(channels=num_channels,
                                samplerate=sample_rate,
                                blocksize=int(sample_rate*rec_duration),
                                callback=sd_callback):
            while True:
                pass


    def sliding_window_chunking(self):
        num_channels = 2
        sample_rate = 16000
        rec_duration = 4
        window = np.zeros(int(5*sample_rate)*1)

        def sd_chunk_callback(rec, frames, time, status):
            rec = rec[:,0]
            rec = np.squeeze(rec)
            window[:len(window)//5] = window[4*(len(window)//5):]
            window[len(window)//5:] = rec
            chunk_pths = self.make_audio_chunks(rec)
            for chunk_pth in chunk_pths:
                self.predict_audio(chunk_pth)

        print(f'Listening...')
        with sd.InputStream(channels=num_channels,
                                samplerate=sample_rate,
                                blocksize=int(sample_rate*rec_duration),
                                callback=sd_chunk_callback):
            while True:
                pass


def main():
    inf = Inference()
    # inf.predict_single_audio(args.wavfile)
    # inf.chunking_audio_prediction(args.wavfile)
    # inf.sliding_window()
    inf.sliding_window_chunking()


if __name__ == '__main__':
    main()
