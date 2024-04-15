# Compressing weights with randomness

This is the code repository for a note published on my website:
<https://tensorial.org/log/compressing-weights-with-randomness>.

The file `mystuff.py` contains a few helpers and is used 
in the two notebooks.
The code is only tested on Ubuntu 22.04 (through WSL2)
on Python 3.10.14.

In order to reproduce, run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

It uses PyTorch but was not tested with GPU support.
A complete run of the notebooks took a good bit less than 
one hour on a system with 8 GB of RAM and an AMD Ryzen 2400G 
(3.2 GHz with 4 core and 8 threads).