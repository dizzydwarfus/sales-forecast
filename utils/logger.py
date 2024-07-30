import logging
from logging import INFO
import os


class MyLogger:
    def __init__(
        self,
        name: str = __name__,
        level: str = "debug",
        log_file: str = "logs.log",
    ):
        # Determine the root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(root_dir, log_file)

        # Initialize logger
        self.logging_level = logging.DEBUG if level == "debug" else logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.logging_level)

        # Check if the self.logger already has handlers to avoid duplicate logging.
        if not self.logger.hasHandlers():
            # Create a file handler
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setLevel(self.logging_level)

            # Create a stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.logging_level)

            # Create a logging format
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            # Add the handlers to the self.logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
