
import numpy
import librosa.display
from matplotlib import pyplot as plt

def main():

    fig, ax = plt.subplots(1, figsize=(20, 5))
    sr = 16000

    S = numpy.load('out.npy').T
    print('S', S.shape)

    librosa.display.specshow(ax=ax, data=S, sr=sr)

    fig.savefig('spec.png')

main()
