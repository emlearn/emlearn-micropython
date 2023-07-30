
# Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from everywhereml.sklearn.ensemble import RandomForestClassifier

# Frameworks to compare
import everywhereml
import emlearn
import m2cgen

def m2c_export(clf, file):

    # XXX: adding support for everywhereml instance
    from m2cgen import assemblers
    from m2cgen.assemblers.ensemble import RandomForestModelAssembler
    assemblers.SUPPORTED_MODELS['everywhereml_RandomForestClassifier'] = RandomForestModelAssembler 

    code = m2cgen.export_to_python(clf, model_name)
    with open(file, 'w') as f:
        f.write(code)

def emlearn_export(clf, file):
    eml = emlearn.convert(clf, kind='RandomForestClassifier')
    eml.save(file=file, name='model', format='csv')

def everywhereml_export(file):

    # TODO: find a way to make everywhereml to work with a standard scikit-learn instance
    #import types
    #from everywhereml.sklearn.ensemble import RandomForestClassifier as RFC
    #ff = RFC()
    #clf.get_template_data = ff.get_template_data
    #clf.to_micropython_file = ff.to_micropython_file
    clf.to_micropython_file(path)


def train():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=7, max_leaf_nodes=20)
    clf.fit(X_train, y_train)

    print('Score: %.2f' % clf.score(X_test, y_test))

    return clf


clf = train()

out_dir = 'digits'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

emlearn_convert(clf, os.path.join(out_dir, 'eml_digits.csv'))
m2c_convert(clf, os.path.join(out_dir, 'm2c_digits.python'))
everywhereml_export(clf, os.path.join(out_dir, 'everywhere_digits.py'))


