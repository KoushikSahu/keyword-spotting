from engine import run
from runtime_config import cls, tfms

if __name__ == '__main__':
    # cls = ['yes', 'no']
    run(cls=cls, tfms=tfms, model_name='dscnn', epochs=1)
