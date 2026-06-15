"""Simple logger setup. Same handler config for every module."""
import logging
import sys


def get_logger(name, log_file=None):
    log = logging.getLogger(name)
    # don't add handlers twice if already configured
    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    log.propagate = False
    return log
