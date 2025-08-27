import picologging as logging
import datetime

# ANSI color codes for priorities
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m", # Red background
    "RESET": "\033[0m"
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Fixed width for aligned priorities
        padded_level = f"{record.levelname:<8}"
        color = COLORS.get(record.levelname, COLORS["RESET"])
        # Short timestamp: yymmdd_hmm
        timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%y%m%d_%H%M")
        return f"{color}{padded_level}{COLORS['RESET']} {timestamp} : {record.getMessage()}"

def setup_colored_logging(level=logging.DEBUG):
    """Configure picologging with colored output and short timestamps for all loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    root_logger = logging.getLogger()  # Root logger affects all loggers by default
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove default handlers if any
    root_logger.addHandler(handler)

# Call once at startup
setup_colored_logging()

logger = logging.getLogger("pkmn_rl")  # Library-scoped root logger
logger.addHandler(logging.NullHandler())  # Prevent "No handlers could be found" warnings
import picologging as logging
import datetime

# ANSI color codes for priorities
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m", # Red background
    "RESET": "\033[0m"
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Fixed width for aligned priorities
        padded_level = f"{record.levelname:<8}"
        color = COLORS.get(record.levelname, COLORS["RESET"])
        # Short timestamp: yymmdd_hmm
        timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%y%m%d_%H%M")
        return f"{color}{padded_level}{COLORS['RESET']} {timestamp} : {record.getMessage()}"

def setup_colored_logging(level=logging.DEBUG):
    """Configure picologging with colored output and short timestamps for all loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    root_logger = logging.getLogger()  # Root logger affects all loggers by default
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove default handlers if any
    root_logger.addHandler(handler)

# Call once at startup
setup_colored_logging()

logger = logging.getLogger("pkmn_rl")  # Library-scoped root logger
logger.addHandler(logging.NullHandler())  # Prevent "No handlers could be found" warnings

