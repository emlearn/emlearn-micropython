
# Manifest is used to include .py files for external C module build
# NOTE: this is a different mechanism than
# Ref https://docs.micropython.org/en/latest/reference/manifest.html
module("emlearn_trees.py", base_path='./emlearn_trees')
module("emlearn_fft.py", base_path='./emlearn_fft')

#include("$(PORT_DIR)/boards/manifest.py")
