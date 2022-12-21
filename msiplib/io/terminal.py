'''
    A collection of several classes and functions that generate or alter output
    to a terminal
'''
import timeit
import datetime

class tfmt:
    """
    ANSI escape sequences for text formatting (tfmt is a shortcut for terminal
    format)

    Should be used in conjunction with colorama to make the output formatting
    work under Windows as well
    """
    # Font formatting
    bold = '\033[1m'
    underline = '\033[4m'
    crossed_out = '\033[9m'

    # Font color
    red = '\033[38;2;235;23;20m'
    orange = '\033[38;2;236;138;30m'
    green = '\033[38;2;63;197;51m'
    yellow = '\033[38;2;242;236;14m'
    blue = '\033[38;2;48;49;239m'
    white = '\033[38;2;255;255;255m'
    black = '\033[38;2;0;0;0m'

    # Background color
    bg_white = '\033[48;2;255;255;255m'
    bg_black = '\033[48;2;0;0;0m'

    # Reset to default
    reset = '\033[0m'


def error(msg):
    """Prints an error message and stops the program"""
    print(tfmt.bold + tfmt.red + "Error:" + tfmt.reset, msg)
    exit()


def warning(msg):
    """Prints a warning message"""
    print(tfmt.bold + tfmt.orange + "Warning:" + tfmt.reset, msg)


def set_terminal_title(title):
    print(f'\33]0;{title}\a', end='', flush=True)


class Timer(object):
    '''Adapted from https://stackoverflow.com/a/50957722'''
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = timeit.default_timer()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.4f seconds' % (
            timeit.default_timer() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, 'a') as file:
                print(str(datetime.datetime.now())+": ", message, file=file)
