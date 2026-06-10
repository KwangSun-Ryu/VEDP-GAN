"""VEDP-GAN GitHub package 통합 CLI."""

import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)


COMMAND_MODULES = {
    "generation": ("generation.run", "main"),
    "prediction": ("prediction.run", "main"),
    "ablation": ("ablation_study.main", "main"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="VEDP-GAN unified runner")
    parser.add_argument("command", choices=sorted(COMMAND_MODULES), help="실행할 작업")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="하위 CLI 인자")
    return parser


def _load_main(module_name, func_name):
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def main():
    parser = build_parser()
    parsed = parser.parse_args()
    module_name, func_name = COMMAND_MODULES[parsed.command]
    sys.argv = [f"{sys.argv[0]} {parsed.command}", *parsed.args]
    return _load_main(module_name, func_name)()


if __name__ == "__main__":
    main()
