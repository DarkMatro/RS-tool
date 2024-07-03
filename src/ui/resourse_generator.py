"""
Paint source icons

classes:
    * ResourceGenerator
"""


import os
import shutil
from pathlib import Path

from ..data.config import get_config


class ResourceGenerator:
    """Generate icons and paint it using theme colors.

    Parameters
    ----------
    primary : str
        name of primary color
    secondary : str
        name of secondary color
    disabled : str
        name of disabled color
    source : Path
         to source folder
    """

    def __init__(self, primary: str, secondary: str, disabled: str, source: Path) -> None:
        """Constructor"""

        self.index = get_config()['material']['path']
        self.contex = [
            (os.path.join(self.index, 'disabled'), disabled),
            (os.path.join(self.index, 'primary'), primary),
        ]
        self.source = source
        self.secondary = secondary

        for folder, _ in self.contex:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)

    def generate(self) -> None:
        """
        Open every icon and change its color
        """
        for icon in os.listdir(self.source):
            if not icon.endswith('.svg'):
                continue
            with open(os.path.join(self.source, icon), 'r', encoding='utf-8') as file_input:
                content_original = file_input.read()

                for folder, color in self.contex:
                    new_content = self.replace_color(content_original, color)
                    new_content = self.replace_color(new_content, self.secondary, '#ff0000')
                    file_to_write = os.path.join(folder, icon)
                    with open(file_to_write, 'w', encoding='utf-8') as file_output:
                        file_output.write(new_content)

    def replace_color(self, content: str, replace: str, color='#0000ff') -> str:
        """
        Replace color in svg text content with 'replace'

        Parameters
        ----------
        content : str
            svg text
        replace : str
            value to replace
        color : str, default = 0000ff

        Returns
        ----------
        out: str
            fixed content
        """
        colors = [color] + [''.join(list(color)[:i] +
                                    ['\\\n'] + list(color)[i:]) for i in range(1, 7)]
        for c in colors:
            content = content.replace(c, replace)

        replace = '#ffffff00'
        color = '#000000'
        colors = [color] + [''.join(list(color)[:i] +
                                    ['\\\n'] + list(color)[i:]) for i in range(1, 7)]
        for c in colors:
            content = content.replace(c, replace)

        return content
