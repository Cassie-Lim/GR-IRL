#!/usr/bin/env python3
import argparse
import pickle
import sys

def load_pickle(filepath):
    """
    Load and return the object from a pickle file.
    WARNING: Unpickling untrusted data can be unsafe.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle file: {filepath}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)

def print_keys_and_shapes(obj):
    """
    Print keys and shapes of values if possible.
    """
    try:
        for key, value in obj.items():
            shape = getattr(value, 'shape', None)
            if shape is not None:
                print(f"{key}: shape = {shape}")
            else:
                print(f"{key}: shape = (not available, type = {type(value).__name__})")
    except AttributeError:
        print("The loaded object is not a dictionary with `.items()` method.")
    except Exception as e:
        print(f"Error while printing keys and shapes: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Load a pickle file and display keys and shape of values."
    )
    parser.add_argument(
        "--pickle_file",
        required=True,
        help="Path to the pickle (.pkl) file to load"
    )
    args = parser.parse_args()

    obj = load_pickle(args.pickle_file)
    print_keys_and_shapes(obj)

if __name__ == "__main__":
    main()
