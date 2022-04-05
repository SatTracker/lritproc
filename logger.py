import datetime
import time


class logger:
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET = '\033[0m'
    COLOR = '\033[1;%dm'
    BOLD = '\033[1m'

    def __init__(self, logfile_path=None, do_print=True):
        if logfile_path is not None:
            self._file = open(logfile_path, 'w')
        else:
            self._file = None
        self._print = True

        self._levels = {
            'DEBUG': False,
            'INFO': True,
            'SUCCESS': True,
            'WARN': True,
            'ERROR': True
        }

        if logfile_path is None and do_print is False:
            print("why are you creating a logger then?! it's not even going to do anything!")

    def _write_message(self, message, level, color):
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self._print and self._levels[level.strip()]:
            print(f'{logger.COLOR.replace("%d", str(logger.GREEN + 30))}{time_str} | {logger.COLOR.replace("%d", str(color + 90))}{level}{logger.COLOR.replace("%d", str(logger.GREEN + 30))} | {logger.COLOR.replace("%d", str(color + 30))}{message}{logger.RESET}')
        if self._file is not None:
            self._file.write(f'{time_str} | {level} | {message}')

    def warn(self, message):
        self._write_message(message, 'WARN   ', logger.YELLOW)

    def error(self, message):
        self._write_message(message, 'ERROR  ', logger.RED)

    def info(self, message):
        self._write_message(message, 'INFO   ', logger.WHITE)

    def success(self, message):
        self._write_message(message, 'SUCCESS', logger.GREEN)

    def debug(self, message):
        self._write_message(message, 'DEBUG  ', logger.CYAN)

    def close(self):
        self._file.close()

    def set_level(self, level: str, enabled: bool):
        if level.upper() not in self._levels:
            raise KeyError("Accepted levels are DEBUG, INFO, WARN, ERROR, SUCCESS")
        self._levels[level.upper()] = enabled
