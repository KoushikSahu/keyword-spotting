from engine import run
from preprocess import Audio, Compose

if __name__ == '__main__':
    # cls = ['yes', 'no']
    cls = ['follow', 'house', 'left', 'marvin', 'nine', 'no', 'right', 'seven', 'six', 'yes']
    tfms = Compose([Audio.load_audio, Audio.lfbe_delta, Audio.to_tensor])
    run(cls=cls, tfms=tfms, model_name='dscnn', epochs=3)

