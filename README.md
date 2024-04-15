# Compressing weights with randomness

This is the code repository for a note published on my website:
<https://tensorial.org/log/compressing-weights-with-randomness>.

The file `mystuff.py` contains a few helpers and is used 
in the three notebooks.
The code is only tested on Ubuntu 22.04 (through WSL2)
on Python 3.10.14.

In order to reproduce, run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
First, run at least the cells for models `model0`, `model6` and 
`model10` in the notebook `models.ipynb` in order to train the final
models. 
Then, run the other two notebooks.

It uses PyTorch but was only tested on a CPU.
A complete run of the notebooks takes about 
one hour on a system with 8 GB of RAM and an AMD Ryzen 2400G 
(3.2 GHz with 4 core and 8 threads).