# Using `pair_style symmetrix/mace`

First, extract your model in `.json` form. You will need a Python environment with
a compatible `mace` module.
```
python <symmetrix-base>/symmetrix/utilities/convert.py my-mace.model 1 8
```
The result should be `my-mace-1-8.json`. This model is only suitable
for LAMMPS simulations involving H and O.

The appropriate LAMMPS `pair` commands are
```
pair_style    symmetrix/mace
pair_coeff    * * my-mace-1-8.json H O
```
where the final `H O` assumes that `H` and `O` correspond to LAMMPS
types `1` and `2`, respectively.
