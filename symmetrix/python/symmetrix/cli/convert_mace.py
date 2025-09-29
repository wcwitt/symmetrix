#!/usr/bin/env python3

from argparse import ArgumentParser

from ..convert_mace import extract_model_data 

def main():
    parser = ArgumentParser()
    parser.add_argument("--species", "-Z", "-s", nargs="+", help="atomic numbers or chemical symbols to extract", default=[])
    parser.add_argument("--head", "-H", help="Head to keep, ignored unless model is multihead. "
                                             "Defaults to first non-PT head, same as mace.tools.script_utils.remove_pt_head")
    parser.add_argument("--output_file", "-o", help="output filename")
    parser.add_argument("model_file", help="torch model file")
    args = parser.parse_args()

    import json
    from pathlib import Path

    if len(Path(args.model_file).suffix) == 0:
        model_name = Path(args.model_file).name
    else:
        model_name = Path(args.model_file).stem

    output = extract_model_data(args.model_file, args.species, args.head)

    ### ----- WRITE JSON -----

    if args.output_file is None:
        args.output_file = model_name + '-' + '-'.join(str(a) for a in sorted(args.species)) + '.json'
    print("WRITING JSON TO", args.output_file)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)
