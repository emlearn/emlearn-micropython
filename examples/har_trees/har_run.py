
import array

import zipfile
import npyfile
import emlearn_trees

def har_load_test_data(path,
        skip_samples=0, limit_samples=None):
 
    n_features = 92

    with zipfile.ZipFile(path) as archive:
        ext = '.npy'
        files = { f.rstrip(ext): f for f in archive.namelist() if f.endswith(ext) }
        #print('archive-files', files.keys())

        X_file = archive.open('X.npy')
        Y_file = archive.open('Y.npy')

        with npyfile.Reader(Y_file) as labels:
            assert len(labels.shape) == 1

            with npyfile.Reader(X_file) as data:
                # Check that data is expected format: sample x features, int16
                shape = data.shape
                assert len(shape) == 2, shape
                assert shape[1] == n_features, shape
                assert data.itemsize == 2
                assert data.typecode == 'h'

                # Number of labels should match number of data items
                assert data.shape[0] == labels.shape[0] 

                # Read data and label for one image at a time
                data_chunk = n_features
                sample_count = 0

                label_chunks = labels.read_data_chunks(1, offset=1*skip_samples)
                data_chunks = data.read_data_chunks(data_chunk, offset=data_chunk*skip_samples)

                for l_arr, arr in zip(label_chunks, data_chunks):

                    yield arr, l_arr

                    sample_count += 1
                    if limit_samples is not None and sample_count > limit_samples:
                        break



def main():

    model = emlearn_trees.new(10, 1000, 10)

    dataset = 'har_uci'
    dataset = 'har_exercise_1'

    model_path = f'{dataset}_trees.csv'

    # Load a CSV file with the model
    with open(model_path, 'r') as f:
        emlearn_trees.load_model(model, f)


    errors = 0
    total = 0
    data_path = f'{dataset}.testdata.npz'
    print('har-run-load', data_path)
    for features, labels in har_load_test_data(data_path):

        assert len(labels) == 1
        label = labels[0]
        result = model.predict(features)
        if result != label:
            errors += 1
        total += 1

        #print(result, label)

    acc = 1.0 - (errors/float(total))
    print(f'har-run-result accuracy={acc*100:.1f}%')

if __name__ == '__main__':
    main()
