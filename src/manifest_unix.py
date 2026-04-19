
# Manifest is used to include .py files for external C module build
# NOTE: this is a different mechanism than
# Ref https://docs.micropython.org/en/latest/reference/manifest.html
module("emlearn_trees.py", base_path='./emlearn_trees')
module("emlearn_kmeans.py", base_path='./emlearn_kmeans')
module("emlearn_cnn_fp32.py", base_path='./emlearn_cnn_fp32')
module("emlearn_fft.py", base_path='./emlearn_fft')
module("emlearn_linreg.py", base_path='./emlearn_linreg')
module("emlearn_logreg.py", base_path='./emlearn_logreg')

#include("$(PORT_DIR)/boards/manifest.py")
