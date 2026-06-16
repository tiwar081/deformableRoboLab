"""CLI entry point for the example demos (not a demo itself).

Dispatches to a named example; see each example script's own docstring for its
motion and run command.

Run: python -m examples <example_name> [options]   (list: python -m examples --list)
"""
from . import main


if __name__ == "__main__":
    main()
