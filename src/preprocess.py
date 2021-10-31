import librosa
import librosa.display
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Audio():
    def __init__(self, path):
        self.data, self.sr = librosa.core.load(path)

    def lfbe_deltadelta(self, plot=False):
        mel_spec = librosa.feature.melspectrogram(self.data,
                sr=self.sr,
                n_mels=13,
                hop_length=160,
                n_fft=480,
                fmin=20,
                fmax=4000)

        log_mel = librosa.core.power_to_db(mel_spec)

        lfbe_del = librosa.feature.delta(log_mel)
        lfbe_deldel = librosa.feature.delta(lfbe_del)
        features = np.vstack([log_mel, lfbe_del, lfbe_deldel])

        if plot:
            librosa.display.specshow(mel_spec, sr=self.sr,
                    x_axis='time', y_axis='mel')
            plt.title('mel spectogram')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
            librosa.display.specshow(log_mel, sr=self.sr,
                    x_axis='time', y_axis='mel')
            plt.title('log mel spectogram')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
            librosa.display.specshow(lfbe_del, sr=self.sr,
                    x_axis='time', y_axis='mel')
            plt.title('log mel delta')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
            librosa.display.specshow(lfbe_deldel, sr=self.sr,
                    x_axis='time', y_axis='mel')
            plt.title('log mel delta delta')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
            librosa.display.specshow(features,
                    sr=self.sr,
                    x_axis='time',
                    y_axis='mel')
            plt.title('mel power spectogram')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
        return features

if __name__ == '__main__':
    audio_path = Path('../data/google_speech_command/go/004ae714_nohash_0.wav')
    a = Audio(audio_path)
    print(a.lfbe_deltadelta(plot=False).shape)

