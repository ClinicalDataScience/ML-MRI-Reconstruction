"""Functions for logging."""
import datetime
import logging
import os
import sys


def set_up_logging(filename: str) -> None:
    """Set up logging."""
    now = datetime.datetime.now()
    path_log = os.path.abspath(os.getcwd()) + '/logs/'
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    filename_log = now.strftime(path_log + filename + '_' + '%d%m%Y_%H:%M:%S.log')
    logging.basicConfig(
        filename=filename_log,
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    filename_err = now.strftime(path_log + filename + '_' + '%d%m%Y_%H:%M:%S.err')
    sys.stderr = open(filename_err, 'w')


def save_to_log(dictionary: dict) -> None:
    """Save dictionary to log file."""
    for arg, value in sorted(dictionary.items()):
        logging.info('%s: %r', arg, value)


def set_up_logging_and_save_args_and_config(filename: str, args, config) -> None:
    """Set up logging and save args and config."""
    set_up_logging(filename)
    save_to_log(vars(args))
    save_to_log(config)
