from argparse import ArgumentParser
import sys


class ArgParser(ArgumentParser):
    """Inherits from ArgumentParser, and used to print helpful message if an error occurs"""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def get_arguments():
    parser = ArgParser()
    # Dataset can be provided via command line
    parser.add_argument("-c", "--config", type=str,
                        help='config file path')
    parser.add_argument("-s", "--statistics", action='store_true',
                        help='whether print statistics of the training dataset')
    parser.add_argument("--sampling_rate", type=float, default=1.0,
                        help='sampling rate for training set')

    return parser.parse_args()


