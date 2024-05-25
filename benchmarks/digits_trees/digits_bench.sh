
MPREMOTE='python ../micropython/tools/mpremote/mpremote.py'

${MPREMOTE} cp m2c_digits.py :
${MPREMOTE} cp eml_digits.csv :
${MPREMOTE} cp everywhere_digits.py :
${MPREMOTE} cp benchmarks/digits_data.py :
${MPREMOTE} cp benchmarks/digits_run.py :

${MPREMOTE} run benchmarks/digits_main.py

