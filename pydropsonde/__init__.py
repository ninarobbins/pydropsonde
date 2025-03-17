import logging
import datetime
import os


# create pydropsonde logger
logger = logging.getLogger("pydropsonde")
logger.setLevel(logging.DEBUG)

# File Handler
fh_info = logging.FileHandler("info.log")
fh_info.setLevel(logging.INFO)

debug_filename = f"debug.{datetime.datetime.now():%Y%m%d%H%M%S}.log"
fh_debug = logging.FileHandler(debug_filename)
fh_debug.setLevel(logging.DEBUG)
os.symlink(debug_filename, debug_filename + ".link")
os.rename(debug_filename + ".link", "debug.latest.log")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# Formatter
log_format = "{asctime}  {levelname:^8s} {name:^20s}: {message}"
formatter = logging.Formatter(log_format, style="{")
fh_info.setFormatter(formatter)
fh_debug.setFormatter(formatter)
ch.setFormatter(formatter)

# Add file and streams handlers to the logger
logger.addHandler(fh_info)
logger.addHandler(fh_debug)
logger.addHandler(ch)
