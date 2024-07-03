import numpy as np


def export_files(item: tuple[str, np.ndarray], path: str) -> None:
    print(path + "/" + item[0])
    np.savetxt(fname=path + "/" + item[0], X=item[1], fmt="%10.5f")
