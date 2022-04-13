import datetime
import platform
import sys


class logger:

    RESET = '\033[0m'
    COLOR = '\033[1;%dm'
    BOLD = '\033[1m'

    _COLORS = {
        '&0': 30,
        '&1': 34,
        '&2': 32,
        '&3': 36,
        '&4': 31,
        '&5': 35,
        '&6': 33,
        '&7': 37,
        '^0': 40,
        '^1': 44,
        '^2': 42,
        '^3': 46,
        '^4': 41,
        '^5': 45,
        '^6': 43,
        '^7': 47,
        '&8': 90,
        '&9': 94,
        '&a': 92,
        '&b': 96,
        '&c': 91,
        '&d': 95,
        '&e': 93,
        '&f': 97,
        '^8': 100,
        '^9': 104,
        '^a': 102,
        '^b': 106,
        '^c': 101,
        '^d': 105,
        '^e': 103,
        '^f': 107,
    }

    _COLORS_NAMES = {
        'BLACK': 30,
        'RED': 31,
        'GREEN': 32,
        'YELLOW': 33,
        'BLUE': 34,
        'MAGENTA': 35,
        'CYAN': 36,
        'WHITE': 37,
        'BRIGHT_BLACK': 90,
        'BRIGHT_RED': 91,
        'BRIGHT_GREEN': 92,
        'BRIGHT_YELLOW': 93,
        'BRIGHT_BLUE': 94,
        'BRIGHT_MAGENTA': 95,
        'BRIGHT_CYAN': 96,
        'BRIGHT_WHITE': 97,
    }

    def __init__(self, path=None, do_print=True):
        if path is not None:
            self._file = open(path, 'a')
        else:
            self._file = None
        self._print = do_print

        self._levels = {
            'DEBUG': False,
            'INFO': True,
            'SUCCESS': True,
            'WARN': True,
            'ERROR': True
        }

    def _write_message(self, message, level, fg, bg='^0'):
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self._print and self._levels[level.strip()]:
            print(self.format_colors(f'&2{bg}{time_str} | {fg}{level:7}&2 | {fg}{message}&r'))
        if self._file is not None:
            self._file.write(self.format_colors(f'{time_str} | {level} | {message}\n', strip=True))

    def warn(self, message):
        self._write_message(message, 'WARN   ', '&e')

    def error(self, message):
        self._write_message(message, 'ERROR  ', '&1')

    def info(self, message):
        self._write_message(message, 'INFO   ', '&f')

    def success(self, message):
        self._write_message(message, 'SUCCESS', '&a')

    def debug(self, message):
        self._write_message(message, 'DEBUG  ', '&3')

    def close(self):
        self._file.close()

    def set_level(self, level: str, enabled: bool):
        if level.upper() not in self._levels:
            raise KeyError("Accepted levels are DEBUG, INFO, WARN, ERROR, SUCCESS")
        self._levels[level.upper()] = enabled

    def set_file(self, path):
        self._file = open(path, 'a')

    @staticmethod
    def format_colors(message: str, strip=False, length=None):
        if '&r' in message:
            message = message.replace('&r', logger.RESET if not strip else '')
        index = 0
        while '&' in message[index:]:
            index = message.find('&', index)
            try:
                message = message.replace(message[index:index + 2], logger.COLOR.replace('%d', str(logger._COLORS[message[index:index + 2]])) if not strip else '')
            except KeyError:
                index += 1
        index = 0
        while '^' in message[index + 1:]:
            index = message.find('^', index)
            try:
                message = message.replace(message[index:index + 2], logger.COLOR.replace('%d', str(logger._COLORS[message[index:index + 2]])) if not strip else '')
            except KeyError:
                index += 1
        if length is not None:
            return message.ljust(length)
        return message

    @staticmethod
    def progress_bar(length, progress, max_progress, prefix, suffix, empty_char=' ', fill_char=' ', bookend=('[', ']'), empty_fg='&f', empty_bg='^f', fill_fg='&a', fill_bg='^a'):
        print(logger.format_colors(f'&2\r{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | &aPROG    &2|&r&f {prefix} {bookend[0]}{fill_fg}{fill_bg}{"".ljust(int((progress / max_progress) * length), empty_char)}{empty_fg}{empty_bg}{"".ljust(length - int((progress / max_progress) * length), fill_char)}&r&f{bookend[1]} {suffix}', length=190), end='')


if __name__ == '__main__':
    from colorama import init
    init()
    print(' ', end='')
    for fg in [*'0123456789abcdef']:
        for bg in [*'0123456789abcdef']:
            print(logger.format_colors(f'&{fg}^{bg}\u2592\u2592&r'), end='')
        print()
