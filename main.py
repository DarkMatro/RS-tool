"""
Starting point.
"""

from multiprocessing import freeze_support

from src.start_program import start_program

if __name__ == "__main__":
    freeze_support()
    start_program()
