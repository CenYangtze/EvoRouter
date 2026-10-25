import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog="python -m evorouter.training")
    parser.add_argument("command", choices=("collect", "sft", "rl"))
    arguments = sys.argv[1:]
    args = parser.parse_args(arguments[:1])
    importlib.import_module(f"evorouter.training.{args.command}").main(arguments[1:])


if __name__ == "__main__":
    main()
