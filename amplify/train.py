import argparse
import amplify


def argParser():
    """
    Args:
    """
    parser = argparse.ArgumentParser(
        description="This project aims to modernize how buildings interact with the grid"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=amplify.__version__),
    )
    return parser.parse_args()


def main():
    p = argParser()
    print("Hello Amplify")


if __name__ == "__main__":
    main()
