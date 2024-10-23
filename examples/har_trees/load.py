
import sys
import pickle

import emlearn


def export_model(path, out):

    with open(path, "rb") as f:
        classifier = pickle.load(f)

        print(classifier)

        classes = classifier.classes_
        class_mapping = dict(zip(classes, range(len(classifier.classes_))))
        print(class_mapping)

        cmodel = emlearn.convert(classifier)
        cmodel.save(name='harmodel', format='csv', file=out)

def main():

    inp, out = sys.argv[1:3]

    export_model(inp, out)
    print('Wrote', out)

if __name__ == '__main__':
    
    main()
