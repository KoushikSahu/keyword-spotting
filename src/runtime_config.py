from preprocess import Audio, Compose

cls = ['yes', 'no']
tfms = Compose([Audio.load_audio, Audio.lfbe_delta, Audio.to_tensor])
