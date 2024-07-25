from logging import error

import numpy as np
import pandas as pd
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QHeaderView
from qtpy.QtCore import QObject, Qt
from seaborn import color_palette, violinplot, boxplot, swarmplot
from sklearn.feature_selection import SelectPercentile

from qfluentwidgets import MessageBox
from src import get_parent, get_config
from src.pandas_tables import PandasModelSmoothedDataset, PandasModelBaselinedDataset, \
    PandasModelDeconvolutedDataset, PandasModelIgnoreDataset, PandasModelDescribeDataset
from src.stages.datasets.classes.undo import CommandDeleteDatasetRow
from src.stages.datasets.functions.stat_test import check_normality, \
    hotelling_t2_with_pca, permutation_test, mann_whitney_u_test, \
    multivariate_bootstrap_comparison


class Datasets(QObject):
    """
    parent is context
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self._ascending_ignore_table = False
        self.reset()
        self._set_ui()
        self.init_current_filename_combobox()

    def reset(self):
        mw = get_parent(self.parent, "MainWindow")
        defaults = get_config('defaults')
        self.reset_smoothed_dataset_table()
        self.reset_baselined_dataset_table()
        self.reset_deconvoluted_dataset_table()
        self.reset_ignore_dataset_table()
        self.reset_describe_dataset_tables()
        self.init_current_filename_combobox()
        mw.ui.select_percentile_spin_box.setValue(defaults["select_percentile_spin_box"])
        mw.ui.violin_describe_plot_widget.canvas.axes.cla()
        mw.ui.violin_describe_plot_widget.canvas.draw()
        mw.ui.boxplot_describe_plot_widget.canvas.axes.cla()
        mw.ui.boxplot_describe_plot_widget.canvas.draw()
        mw.ui.bootstrap_plot_widget.canvas.axes.cla()
        mw.ui.bootstrap_plot_widget.canvas.draw()
        mw.ui.stat_test_text_edit.setText('')

    def _reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        -------
        event: QMouseEvent

        field_id: str
            name of field
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        mw = get_parent(self.parent, "MainWindow")
        value = get_config('defaults')[field_id]
        match field_id:
            case 'select_percentile_spin_box':
                mw.ui.select_percentile_spin_box.setValue(value)
            case _:
                return

    def _set_ui(self):
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        self._initial_smoothed_dataset_table()
        self._initial_baselined_dataset_table()
        self._initial_deconvoluted_dataset_table()
        self._initial_ignore_dataset_table()
        mw.ui.check_all_push_button.clicked.connect(
            lambda: mw.ui.ignore_dataset_table_view.model().set_checked({})
        )
        mw.ui.select_percentile_spin_box.valueChanged.connect(context.set_modified)
        mw.ui.select_percentile_spin_box.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'select_percentile_spin_box'))
        mw.ui.select_percentile_push_button.clicked.connect(
            self.feature_select_percentile
        )
        self._initial_describe_dataset_tables()
        mw.ui.describe_1_SpinBox.valueChanged.connect(
            lambda: mw.ui.describe_1_SpinBox.setMaximum(self.parent.group_table.rowCount))
        mw.ui.describe_2_SpinBox.valueChanged.connect(
            lambda: mw.ui.describe_2_SpinBox.setMaximum(self.parent.group_table.rowCount))
        mw.ui.update_describe_push_button.clicked.connect(self.update_describe_tables)
        mw.ui.violin_box_plots_update_push_button.clicked.connect(self.update_violin_boxplot)
        mw.ui.stat_test_btn.clicked.connect(self.stat_test)

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dt: dict
            all class attributes data
        """
        mw = get_parent(self.parent, "MainWindow")
        dt = {"smoothed_dataset_df": mw.ui.smoothed_dataset_table_view.model().dataframe(),
              "baselined_dataset_df": mw.ui.baselined_dataset_table_view.model().dataframe(),
              "deconvoluted_dataset_df": mw.ui.deconvoluted_dataset_table_view.model().dataframe(),
              "ignore_dataset_df": mw.ui.ignore_dataset_table_view.model().dataframe(),
              "IgnoreTableChecked": mw.ui.ignore_dataset_table_view.model().checked,
              "select_percentile_spin_box": mw.ui.select_percentile_spin_box.value(),
              }
        return dt

    def load(self, db: dict):
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.smoothed_dataset_table_view.model().set_dataframe(db["smoothed_dataset_df"])
        mw.ui.baselined_dataset_table_view.model().set_dataframe(db["baselined_dataset_df"])
        mw.ui.deconvoluted_dataset_table_view.model().set_dataframe(db["deconvoluted_dataset_df"])
        mw.ui.ignore_dataset_table_view.model().set_checked(db["IgnoreTableChecked"])
        mw.ui.ignore_dataset_table_view.model().set_dataframe(db["ignore_dataset_df"])
        mw.ui.select_percentile_spin_box.setValue(db["select_percentile_spin_box"])
        self.init_current_filename_combobox()

    def _initial_smoothed_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.smoothed_dataset_table_view.verticalScrollBar().valueChanged.connect(
            mw.move_side_scrollbar
        )
        mw.ui.smoothed_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _initial_baselined_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.baselined_dataset_table_view.verticalScrollBar().valueChanged.connect(
            mw.move_side_scrollbar)
        mw.ui.baselined_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _initial_deconvoluted_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.deconvoluted_dataset_table_view.verticalScrollBar().valueChanged.connect(
            mw.move_side_scrollbar)
        mw.ui.deconvoluted_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mw.ui.deconvoluted_dataset_table_view.keyPressEvent = self.decomp_table_key_pressed
        mw.ui.deconvoluted_dataset_table_view.model().modelReset.connect(
            self.init_current_filename_combobox)

    def _initial_ignore_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.ignore_dataset_table_view.verticalScrollBar().valueChanged.connect(
            mw.move_side_scrollbar)
        mw.ui.ignore_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mw.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(0, 220)
        mw.ui.ignore_dataset_table_view.setSortingEnabled(True)
        mw.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(1, 200)
        mw.ui.ignore_dataset_table_view.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        mw.ui.ignore_dataset_table_view.horizontalHeader().resizeSection(2, 200)
        mw.ui.ignore_dataset_table_view.horizontalHeader().sectionClicked.connect(
            self._ignore_dataset_table_header_clicked)

    def _initial_describe_dataset_tables(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.describe_dataset_table_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mw.ui.describe_2nd_group.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mw.ui.describe_1st_group.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def reset_smoothed_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame(columns=["Class", "Filename"])
        model = PandasModelSmoothedDataset(mw, df)
        mw.ui.smoothed_dataset_table_view.setModel(model)

    def reset_baselined_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame(columns=["Class", "Filename"])
        model = PandasModelBaselinedDataset(mw, df)
        mw.ui.baselined_dataset_table_view.setModel(model)

    def reset_deconvoluted_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame(columns=["Class", "Filename"])
        model = PandasModelDeconvolutedDataset(mw, df)
        model.modelReset.connect(
            lambda: self.parent.ml.dataset_type_cb_current_text_changed('Decomposed'))
        mw.ui.deconvoluted_dataset_table_view.setModel(model)

    def reset_ignore_dataset_table(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame(columns=["Feature", "Score", "P value"])
        model = PandasModelIgnoreDataset(mw, df, {})
        model.modelReset.connect(
            lambda: self.parent.ml.dataset_type_cb_current_text_changed('Decomposed'))
        mw.ui.ignore_dataset_table_view.setModel(model)

    def reset_describe_dataset_tables(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        model = PandasModelDescribeDataset(mw, pd.DataFrame())
        mw.ui.describe_dataset_table_view.setModel(model)

        model = PandasModelDescribeDataset(mw, pd.DataFrame())
        mw.ui.describe_1st_group.setModel(model)
        model = PandasModelDescribeDataset(mw, pd.DataFrame())
        mw.ui.describe_2nd_group.setModel(model)

    def init_current_filename_combobox(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.current_filename_combobox.clear()
        if not mw.ui.deconvoluted_dataset_table_view.model():
            return
        q_res = mw.ui.deconvoluted_dataset_table_view.model().dataframe()
        mw.ui.current_filename_combobox.addItem(None)
        mw.ui.current_filename_combobox.addItems(q_res["Filename"])

    def decomp_table_key_pressed(self, key_event) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if (key_event.key() == Qt.Key.Key_Delete
                and mw.ui.deconvoluted_dataset_table_view.selectionModel().currentIndex().row() > -1
                and len(mw.ui.deconvoluted_dataset_table_view.selectionModel().selectedIndexes())):
            context = get_parent(self.parent, "Context")
            command = CommandDeleteDatasetRow(None, context, text="Delete row",
                                              **{'stage': self,
                                                 'dataset_table':
                                                     mw.ui.deconvoluted_dataset_table_view})
            context.undo_stack.push(command)

    def _ignore_dataset_table_header_clicked(self, idx: int):
        mw = get_parent(self.parent, "MainWindow")
        df = mw.ui.ignore_dataset_table_view.model().dataframe()
        column_names = df.columns.values.tolist()
        current_name = column_names[idx]
        self._ascending_ignore_table = not self._ascending_ignore_table
        mw.ui.ignore_dataset_table_view.model().sort_values(
            current_name, self._ascending_ignore_table
        )

    def feature_select_percentile(self) -> None:
        """
        1. Check on all features in the ignore_dataset_table_view
        2. Use VarianceThreshold to find not important features
        3. Uncheck not important features

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        # 1. Check on all features in the ignore_dataset_table_view
        mw.ui.ignore_dataset_table_view.model().set_all_features_checked()

        # 2. Use VarianceThreshold to find not important features
        df = mw.ui.deconvoluted_dataset_table_view.model().dataframe()
        x = df.iloc[:, 2:]
        y = df['Class']
        percentile = mw.ui.select_percentile_spin_box.value()
        selector = SelectPercentile(percentile=percentile)
        selector = selector.fit(x, y)
        feature_names_in = selector.feature_names_in_
        support = selector.get_support()

        # 3. Uncheck not important features
        checked = {}
        for feature_name, b in zip(feature_names_in, support):
            checked[feature_name] = b
        mw.ui.ignore_dataset_table_view.model().set_checked(checked)
        mw.ui.ignore_dataset_table_view.model().set_column_data('Score', selector.scores_)
        mw.ui.ignore_dataset_table_view.model().set_column_data('P value', selector.pvalues_)

    def update_describe_tables(self) -> None:
        """
        1. Update describe_dataset_table_view
        2. Update describe_1st_group
        3. Update describe_2nd_group

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        # 1. Update describe_dataset_table_view
        x, _, _, _, _ = mw.stat_analysis_logic.dataset_for_ml()
        if x.empty:
            return
        df = x.describe()
        mw.ui.describe_dataset_table_view.model().set_dataframe(df)
        # 2. Update describe_1st_group
        group_id_1 = mw.ui.describe_1_SpinBox.value()
        df = mw.ui.deconvoluted_dataset_table_view.model().dataframe()
        ignored_features = mw.ui.ignore_dataset_table_view.model().ignored_features
        df = df.drop(ignored_features, axis=1)
        df2 = df[df['Class'] == group_id_1].describe().iloc[:, 1:]
        mw.ui.describe_1st_group.model().set_dataframe(df2)

        # 3. Update describe_2nd_group
        group_id_2 = mw.ui.describe_2_SpinBox.value()
        df3 = df[df['Class'] == group_id_2].describe().iloc[:, 1:]
        mw.ui.describe_2nd_group.model().set_dataframe(df3)

    def update_violin_boxplot(self) -> None:
        """
        1. Build violin plot for decomposed only data
        2. Build boxplot
        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.ui.deconvoluted_dataset_table_view.model().rowCount() == 0:
            MessageBox('Update violin and box plot failed..', 'No decomposed lines data', mw,
                       {'Ok'})
            return
        # Build dataframe
        df = mw.ui.deconvoluted_dataset_table_view.model().dataframe()
        ignored_features = mw.ui.ignore_dataset_table_view.model().ignored_features
        df = df.drop(ignored_features, axis=1)
        n_rows = df.shape[0]
        col_features, col_value, col_classes, col_filenames = [], [], [], []
        for col in df.columns[2:]:
            col_features.append([col] * n_rows)
            col_value.append(list(df[col].values))
            col_classes.append(df['Class'].to_list())
            col_filenames.append(df['Filename'].to_list())
        col_features = np.array(col_features).flatten()
        col_value = np.array(col_value).flatten()
        col_classes = np.array(col_classes).flatten()
        col_filenames = np.array(col_filenames).flatten()
        df_new = pd.DataFrame({'Class': col_classes, 'Feature': col_features, 'Value': col_value,
                               'Filename': col_filenames})

        # Update violin plot
        self.build_violin_box_plot(df_new)
        # Update Box-plot
        self.build_violin_box_plot(df_new, False)

    def build_violin_box_plot(self, df: pd.DataFrame, violin: bool = True) -> None:
        """
        Grouped violin plots with split violins
        Parameters
        ----------
        df: DataFrame
        violin: bool
            violin or box plot
        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.violin_describe_plot_widget if violin \
            else mw.ui.boxplot_describe_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        palette = color_palette(self.parent.group_table.table_widget.model().groups_colors)
        order = mw.ui.ignore_dataset_table_view.model().features_by_order()
        if violin:
            ax = violinplot(data=df, x='Feature', y='Value', hue='Class', order=order,
                            split=True, inner="quart", fill=False, palette=palette, ax=ax)
        else:
            try:
                ax = boxplot(data=df, x='Feature', y='Value', hue='Class', order=order, fill=False,
                             palette=palette, ax=ax)
            except UnboundLocalError as err:
                error(err)
        cur_filename = mw.ui.current_filename_combobox.currentText()
        if cur_filename is not None:
            ax2 = ax.twinx()
            swarmplot(data=df[df['Filename'] == cur_filename], x='Feature', y='Value', color='red',
                      marker='X',
                      order=order, size=10, ax=ax)
            ax2.set_yticklabels([])
            ax2.set_ylabel('')
            ax2.set_ylim(ax.get_ylim())
            ax.set_title(cur_filename)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError as err:
            error(err)

    def stat_test(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        x, y, _, _, _ = mw.stat_analysis_logic.dataset_for_ml()
        group_id0 = mw.ui.describe_1_SpinBox.value()
        group_id1 = mw.ui.describe_2_SpinBox.value()
        df_0 = x[y.isin([group_id0])]
        df_1 = x[y.isin([group_id1])]
        check_normal_0 = check_normality(df_0)
        check_normal_1 = check_normality(df_1)
        text = ''
        group_name_0 = self.parent.group_table.table_widget.model().get_group_name_by_int(group_id0)
        group_name_1 = self.parent.group_table.table_widget.model().get_group_name_by_int(group_id1)
        for res, group_name in [(check_normal_1, group_name_0), (check_normal_0, group_name_1)]:
            text += (f"Distribution for class = {group_name} is {'' if res[0] else 'not'} normal,"
                     f" p_value = {res[1]}") + '\n'
        if all([check_normal_0[0], check_normal_1[0]]):
            res = hotelling_t2_with_pca(df_0, df_1,
                                        n_components=10 if x.shape[1] >= 10 else x.shape[1])
            text += f"Hotelling's T-squared statistic: {res[0]}" + '\n'
            text += f"P-value: {res[1]}" + '\n'
            text += (f"Are the samples significantly different? {'Yes' if res[2] else 'No'}"
                     + '\n')
        else:
            res = permutation_test(df_0, df_1)
            text += (f"Permutation test statistic: {res[0]}" + '\n' + f"P-value: {res[1]}" + '\n'
                     + f"Are the samples significantly different? {'Yes' if res[2] else 'No'}"
                     + '\n' + '\n')
            res = mann_whitney_u_test(df_0, df_1)
            text += ("Mann-Whitney U Test Results:" + '\n' + res[0].to_string() + '\n'
                     + f"Are the samples significantly different in any feature? "
                       f"{'Yes' if res[1] else 'No'}" + '\n' + '\n')
        res = multivariate_bootstrap_comparison(
            df_0, df_1, mw.ui.bootstrap_plot_widget)
        text += ("Multivariate bootstrap comparison:" + '\n' f"Observed Test Statistic: {res[0]}"
                 + '\n' + f"P-value: {res[1]}" + '\n'
                 + f"Are Samples Different? {'Yes' if res[2] else 'No'}" + '\n' + '\n')
        mw.ui.stat_test_text_edit.setText(text)
