# utils/logger.py

import logging
import os

def setup_logger(log_file):
    """
    Sets up the logger to output logs to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Extract the directory from log_file
    log_dir = os.path.dirname(log_file)

    if log_dir:
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Log directory created or already exists: {log_dir}")

    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Avoid adding multiple handlers if logger already has handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info("Logger has been set up successfully.")

    return logger