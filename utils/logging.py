import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger named 'name'"""
    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.setLevel(logging.INFO)

    return logging.getLogger('data')
