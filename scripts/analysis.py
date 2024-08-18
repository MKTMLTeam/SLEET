import sys
import os

sys.path.append(os.getcwd())

from src.analyzer import analysis


if __name__ == '__main__':
    analysis()
