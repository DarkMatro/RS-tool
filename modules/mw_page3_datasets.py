import numpy as np
from seaborn import violinplot, swarmplot, color_palette, boxplot
from qfluentwidgets import MessageBox
from pandas import DataFrame


class DatasetsManager:

    def __init__(self, parent):
        self.parent = parent

    def update_describe_tables(self) -> None:
        """
        1. Update describe_dataset_table_view
        2. Update describe_1st_group
        3. Update describe_2nd_group

        Returns
        -------
        None
        """
        # 1. Update describe_dataset_table_view
        X, _, _, _, _ = self.parent.stat_analysis_logic.dataset_for_ml()
        if X.empty:
            return
        df = X.describe()
        self.parent.ui.describe_dataset_table_view.model().set_dataframe(df)
        # 2. Update describe_1st_group
        group_id_1 = self.parent.ui.describe_1_SpinBox.value()
        df = self.parent.ui.deconvoluted_dataset_table_view.model().dataframe()
        ignored_features = self.parent.ui.ignore_dataset_table_view.model().ignored_features
        df = df.drop(ignored_features, axis=1)
        df2 = df[df['Class'] == group_id_1].describe().iloc[:, 1:]
        self.parent.ui.describe_1st_group.model().set_dataframe(df2)

        # 3. Update describe_2nd_group
        group_id_2 = self.parent.ui.describe_2_SpinBox.value()
        df3 = df[df['Class'] == group_id_2].describe().iloc[:, 1:]
        self.parent.ui.describe_2nd_group.model().set_dataframe(df3)

    def update_violin_boxplot(self) -> None:
        """
        1. Build violin plot for decomposed only data
        2. Build boxplot
        Returns
        -------
        None
        """
        if self.parent.ui.deconvoluted_dataset_table_view.model().rowCount() == 0:
            MessageBox('Update violin and box plot failed..', 'No decomposed lines data', self.parent, {'Ok'})
            return
        # Build dataframe
        df = self.parent.ui.deconvoluted_dataset_table_view.model().dataframe()
        ignored_features = self.parent.ui.ignore_dataset_table_view.model().ignored_features
        df = df.drop(ignored_features, axis=1)
        n_rows = df.shape[0]
        col_features = []
        col_value = []
        col_classes = []
        col_filenames = []
        for col in df.columns[2:]:
            col_features.append([col] * n_rows)
            col_value.append(list(df[col].values))
            col_classes.append(list(df['Class'].values))
            col_filenames.append(list(df['Filename'].values))
        col_features = np.array(col_features).flatten()
        col_value = np.array(col_value).flatten()
        col_classes = np.array(col_classes).flatten()
        col_filenames = np.array(col_filenames).flatten()
        df_new = DataFrame({'Class': col_classes, 'Feature': col_features, 'Value': col_value,
                            'Filename': col_filenames})
        # Update violin plot
        self.build_violin_box_plot(df_new, True)
        # Update Box-plot
        self.build_violin_box_plot(df_new, False)

    def build_violin_box_plot(self, df: DataFrame, violin: bool = True) -> None:
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

        plot_widget = self.parent.ui.violin_describe_plot_widget if violin \
            else self.parent.ui.boxplot_describe_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        palette = color_palette(self.parent.ui.GroupsTable.model().groups_colors)
        order = self.parent.ui.ignore_dataset_table_view.model().features_by_order()
        if violin:
            vp = violinplot(data=df, x='Feature', y='Value', hue='Class', order=order,
                            split=True, inner="quart", fill=False, palette=palette, ax=ax)
        else:
            vp = boxplot(data=df, x='Feature', y='Value', hue='Class', order=order, fill=False, palette=palette, ax=ax)
        cur_filename = self.parent.ui.current_filename_combobox.currentText()
        if cur_filename is not None:
            ax2 = ax.twinx()
            swarmplot(data=df[df['Filename'] == cur_filename], x='Feature', y='Value', color='red', marker='X',
                      order=order, size=10, ax=ax)
            ax2.set_yticklabels([])
            ax2.set_ylabel('')
            ax2.set_ylim(vp.get_ylim())
            ax.set_title(cur_filename)
        try:
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
