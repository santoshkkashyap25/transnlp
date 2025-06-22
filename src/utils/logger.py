import logging
from src.config import LOG_FILE_PATH

def setup_logging():
    """Configures the application-wide logger."""
    logging.basicConfig(
        level=logging.DEBUG, # CHANGE THIS LINE from logging.INFO to logging.DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("StreamlitAppLogger")

# Initialize logger for other modules to import
app_logger = setup_logging()