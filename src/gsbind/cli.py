"""Small command-line interface for the inference release."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="GSbind: protein interface prediction from structure")
    parser.add_argument("--version", action="version", version="GSbind 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict one protein chain from PDB or mmCIF")
    predict.add_argument("--structure", type=Path, required=True)
    predict.add_argument("--chain", help="Chain identifier; required for structures with multiple protein chains")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    predict.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N (default: cpu)")
    predict.add_argument("--threads", type=int, default=2)
    predict.add_argument("--save-hidden", action="store_true", help="Also save final 128-dimensional residue features")
    args = parser.parse_args()
    from .inference import predict as run
    try:
        result = run(args.structure, args.checkpoint, args.output, args.chain, args.device, args.save_hidden, args.threads)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"gsbind: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
