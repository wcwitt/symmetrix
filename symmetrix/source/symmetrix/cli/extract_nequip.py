#!/usr/bin/env python3

from argparse import ArgumentParser

from ..extract_nequip_data import extract_nequip_data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="NequIP package (.nequip.zip), checkpoint (.ckpt), "
        "or nequip.net model ID (nequip.net:group-name/model-name:version).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--atomic-numbers",
        "-Z",
        "-z",
        nargs="+",
        help="Atomic numbers to extract.",
        default=[],
    )
    group.add_argument(
        "--chemical-symbols",
        "-s",
        nargs="+",
        help="Chemical symbols to extract.",
        default=[],
    )
    parser.add_argument(
        "--num-spline-points",
        type=int,
        default=256,
        help="Number of spline points.",
    )
    parser.add_argument("--output", "-o", help="Output filename.")
    args = parser.parse_args()

    import json
    from pathlib import Path
    from importlib.metadata import version

    if args.model.startswith("nequip.net:"):
        model_name = args.model.split("/")[-1].replace(":", "-")
    else:
        model_name = Path(args.model).name
        for suffix in [".nequip.zip", ".zip", ".ckpt"]:
            if model_name.endswith(suffix):
                model_name = model_name[: -len(suffix)]
                break

    species = (
        args.atomic_numbers if args.chemical_symbols == [] else args.chemical_symbols
    )
    output = extract_nequip_data(
        args.model, species, num_spline_points=args.num_spline_points
    )
    output["extracted_with"] = {p: version(p) for p in ("symmetrix", "nequip", "torch")}

    ### ----- WRITE JSON -----

    if args.output is None:
        args.output = (
            model_name + "-" + "-".join(str(a) for a in sorted(species)) + ".json"
        )
    print("WRITING JSON TO", args.output)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)
