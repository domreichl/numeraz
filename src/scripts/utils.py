from argparse import ArgumentParser, Namespace


def parse_args(args: list) -> dict:
    parser = ArgumentParser()
    for arg in args:
        parser.add_argument(f"--{arg}", type=str, required=True)
    args: Namespace = parser.parse_args()
    return vars(args)
