from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from subprocess import run

def init_argparse() -> ArgumentParser:
    parser = ArgumentParser(
        description="Combine collected profraw files into profdata for llvm-cov.",
    )
    parser.add_argument(
        "--profile-dir",
        help="Directory to recursively find profiles in.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--llvm-profdata-tool",
        help="llvm-prodata binary.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output",
        help="Output profdata file",
        required=True,
        type=Path,
    )
    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    profiles = glob("**/*.profraw", root_dir=args.profile_dir, recursive=True)
    profiles = list(map(str,map(lambda v: Path(args.profile_dir)/v ,profiles)))
    profiles = " ".join(profiles)

    cmd = f"{args.llvm_profdata_tool} merge {profiles} -o {args.output}"
    run(args=cmd,shell=True)


if __name__ == "__main__":
    main()
