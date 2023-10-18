from pathlib import Path


def recent_files_exists_and_empty():
    path = 'recent-files.txt'
    return Path(path).exists() and Path(path).stat().st_size == 0
