from . import pipeline as pi
import argparse
from importlib.metadata import version

__version__ = version("pydropsonde")


def main():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    parser.add_argument(
        "-c",
        "--config_file_path",
        default="./dropsonde.cfg",
        help="config file path for pydropsonde, "
        + "by default the config file is dropsonde.cfg in the current directory."
        + "Otherwise path to directory and filename need to be defined",
        required=False,
    )

    args = parser.parse_args()
    if args.config_file_path:
        import os

        config_file_path = args.config_file_path
        config_dirname = os.path.dirname(config_file_path)

        # check if given config file directory exists
        if not os.path.exists(config_dirname):
            raise FileNotFoundError(f"Directory {config_dirname} not found.")
        else:
            # check if config file exists inside
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(
                    f"File {config_file_path} does not exist. Please check file name."
                )
            else:
                import configparser

                config = configparser.ConfigParser()
                config.optionxform = str
                config.read(config_file_path)

        pi.run_pipeline(pi.pipeline, config)
