import logging
import sys

from config import OUT_PATH

LOG_PATH = OUT_PATH / "logs"


def get_logger(script_path: str, timestamp: str) -> logging.Logger:
    """
    Returns logger that logs to stdout and unique file at the same time.

    Args:
        script_path: path to script that creates logs.
        timestamp: text that differentiates created files between runs.

    Returns:
        Logger object.
    """
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    script_name = script_path.split("/")[-1]
    log_filename = f'{script_name.split(".")[0]}_{timestamp}.log'

    root_logger = logging.getLogger()
    root_logger.setLevel("INFO")

    file_handler = logging.FileHandler(LOG_PATH / log_filename)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(stream_handler)

    return root_logger
