#!/usr/bin/env python3

from argparse import ArgumentParser

from ..extract_mace_data import extract_mace_data 

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Torch model file.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--atomic-numbers", "-Z", "-z", nargs="+", help="Atomic numbers to extract.", default=[])
    group.add_argument("--chemical-symbols", "-s", nargs="+", help="Chemical symbols to extract.", default=[])
    parser.add_argument("--head", "-H", help="Head to keep, ignored unless model is multihead. "
                                             "Defaults to first non-PT head, same as mace.tools.script_utils.remove_pt_head")
    parser.add_argument("--output", "-o", help="Output filename.")
    args = parser.parse_args()

    import json
    from pathlib import Path

    if len(Path(args.model).suffix) == 0:
        model_name = Path(args.model).name
    else:
        model_name = Path(args.model).stem

    species = args.atomic_numbers if args.chemical_symbols == [] else args.chemical_symbols
    output = extract_mace_data(args.model, species, args.head)

    ### ----- WRITE JSON -----

    if args.output is None:
        args.output = model_name + '-' + '-'.join(str(a) for a in sorted(species)) + '.json'
    print("WRITING JSON TO", args.output)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)
