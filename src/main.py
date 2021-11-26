from engine import run
from preprocess import Audio, Compose

if __name__ == '__main__':
  # cls = ['yes', 'no']
  cls = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
  tfms = Compose([Audio.load_audio, Audio.lfbe_delta, Audio.to_tensor])
  run(cls=cls, tfms=tfms, model_name='lstm', epochs=50)
