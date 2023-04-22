from pathlib import Path

import modules

qt_material_path = Path(modules.__file__).parent

fonts_path = qt_material_path / "fonts"
datas = [(str(fonts_path), "qt_material/fonts")]

themes_path = qt_material_path / "themes"
datas += [(str(themes_path), "qt_material/themes")]

dock_path = qt_material_path / "dock_theme.ui"
datas += [(str(dock_path), "qt_material")]

template_path = qt_material_path / "material.css.template"
datas += [(str(template_path), "qt_material")]

resources_path = qt_material_path / "resources"
datas += [(str(resources_path), "qt_material/resources")]
