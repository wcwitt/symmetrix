# `symmetrix`

To build the `symmetrix` Python package:

```
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/symmetrix
pip install .
```

### Generating Symmetrix `.json` model files

Once the Python package is installed, use
```
symmetrix_extract_mace my-mace.model --atomic-numbers 1 8
```
from the command line to extract a `.json` file from a Torch-based model.
The result will be `my-mace-1-8.json`, and this model is only suitable
for simulations involving H and O.

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
See [the source code](source/symmetrix/symmetrix_calc.py) and [this test](test/test_symmetrix_calc.py)
for additional details.
