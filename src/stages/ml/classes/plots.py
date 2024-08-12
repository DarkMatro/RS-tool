# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module defines the `Plots` class for managing and updating various ML plots
in the graphical user interface.

The `Plots` class includes methods for creating and refreshing decision score plots,
confusion matrices, ROC and precision-recall curves, and more, based on the selected
classifier and the current state of the data.
"""
import re
from logging import error
from os import environ

import numpy as np
import optuna
from optuna.visualization import _contour
import pandas as pd
import xgboost
from qtpy.QtCore import QObject
from joblib import parallel_backend
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from seaborn import color_palette, histplot
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay, permutation_importance, \
    DecisionBoundaryDisplay
from sklearn.metrics import DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay, \
    precision_recall_curve, average_precision_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import LearningCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree

from src import get_parent
from src.data.default_values import get_optuna_params
from src.stages.ml.functions.hyperopt import get_study


class Plots(QObject):
    """
    A class for managing and updating ML-related plots in the graphical user interface.

    This class is responsible for creating and refreshing various plots, such as
    decision score plots, confusion matrices, ROC curves, precision-recall curves,
    and feature importance plots, based on the type of classifier and the available data.

    Parameters
    ----------
    parent : ML stage
        The parent stage to which this backend is attached.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the ML plotting backend.

        Parameters
        ----------
        parent : ML stage
            The parent stage to which this backend is attached.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.parent = parent

    @property
    def data(self):
        """
        Data property of parent.
        """
        return self.parent.data

    def _get_plot_colors(self, classes: list[int]) -> list[str]:
        """
        Retrieve a list of plot colors corresponding to the given classes.

        Parameters
        ----------
        classes : list[int]
            List of class identifiers.

        Returns
        -------
        list[str]
            List of color names corresponding to the classes.
        """
        mw = get_parent(self.parent, "MainWindow")
        plt_colors = []
        for cls in classes:
            clr = mw.context.group_table.get_color_by_group_number(cls)
            color_name = clr.name() if mw.ui.sun_Btn.isChecked() else clr.lighter().name()
            plt_colors.append(color_name)
        return plt_colors

    def update_plots(self, cl_type: str) -> None:
        """
        Update all plots related to the specified classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to update the plots.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        # page 1 of fit results
        if cl_type in ['LDA', 'Logistic regression', 'SVC']:
            self._build_decision_score_plot(cl_type)
        else:
            mw.ui.decision_score_plot_widget.canvas.gca().cla()
            mw.ui.decision_score_plot_widget.canvas.gca().text(.5, .5, 'No data', color='red')
            mw.ui.decision_score_plot_widget.canvas.draw()
        self._update_scores_plot(cl_type)
        self._update_dm_plot(cl_type)
        binary = len(self.data[cl_type]['model'].classes_) == 2
        if binary:
            self._update_roc_plot_bin(cl_type)
            self._update_pr_plot_bin(cl_type)
        else:
            self._update_roc_plot(cl_type)
            self._update_pr_plot(cl_type)
        # page 2 of fit results
        self._build_permutation_importance_plot(cl_type, True)
        self._build_permutation_importance_plot(cl_type)
        self.update_plot_tree(cl_type)
        if cl_type in ['LDA', 'Logistic regression', 'SVC']:
            self._update_features_plot(cl_type)
        elif cl_type in ['Decision Tree', 'Random Forest']:
            self._update_features_plot_random_forest(cl_type)
        else:
            plot_widget = mw.ui.features_plot_widget
            plot_widget.canvas.gca().cla()
            plot_widget.canvas.gca().text(.5, .5, 'No data', color='red')
            plot_widget.canvas.draw()
            try:
                plot_widget.canvas.draw()
            except ValueError:
                pass
        if cl_type == 'XGBoost':
            if self.data[cl_type]['dataset_type'] == 'Decomposed':
                self._update_features_plot_xgboost()
            else:
                self._update_features_plot_random_forest(cl_type)
            self.update_xgboost_tree_plot()
        # page 3 of fit results
        if binary:
            self._build_calibration_plot(cl_type)
            self._build_dev_curve_plot(cl_type)

    def _build_decision_score_plot(self, cl_type) -> None:
        """
        Build and update the decision score plot for the given classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to build the decision score plot.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.decision_score_plot_widget
        context = get_parent(self.parent, "Context")
        ax = plot_widget.canvas.gca()
        ax.cla()
        y_score_decision = self.data[cl_type]['model'].decision_function(self.data[cl_type]['x'])
        class_names = self.data[cl_type]['target_names']
        y = self.data[cl_type]['y']
        if len(y_score_decision.shape) > 1 and y_score_decision.shape[1] > 1:
            decision_function_values_1d = np.array(
                [v_row[y[i] - 1] for i, v_row in enumerate(y_score_decision)])
        else:
            decision_function_values_1d = y_score_decision
        palette = color_palette(context.group_table.table_widget.model().groups_colors)
        c_n = [class_names[i - 1] if i <= len(class_names) else 'NoName' for i in y]
        df = pd.DataFrame({'Decision score value': decision_function_values_1d, 'label': c_n})
        histplot(data=df, x='Decision score value', hue='label', palette=palette, kde=True, ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_scores_plot(self, cl_type: str) -> None:
        """
        Update the decision boundary plot for the given classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to update the decision boundary plot.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.decision_boundary_plot_widget
        context = get_parent(self.parent, "Context")
        plot_widget.canvas.gca().cla()
        features_2d = self.data[cl_type]['features_2d']
        if len(features_2d.shape) <= 1 or features_2d.shape[1] <= 1:
            plot_widget.canvas.gca().text(.5, .5, 'No data', color='red')
            plot_widget.canvas.draw()
            return
        dr_method = 'LD' if cl_type == 'LDA' else 'PC'
        evr = self.data[cl_type].get('explained_variance_ratio', [0, 0])
        classes = self.data[cl_type]['model'].classes_
        plt_colors = [context.group_table.get_color_by_group_number(cls).lighter(140).name()
                      for cls in classes]
        y = np.concatenate((self.data[cl_type]['y_train'], self.data[cl_type]['y_test']))
        tp = y == self.data[cl_type]['y_pred_2d']  # True Positive
        DecisionBoundaryDisplay.from_estimator(self.data[cl_type]['model_2d'], features_2d,
                                               grid_resolution=1000,
                                               eps=.5, antialiased=True,
                                               cmap=LinearSegmentedColormap(
                                                   '', None).from_list('', plt_colors),
                                               ax=plot_widget.canvas.gca())
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P',
                   '*']
        for cls in classes:
            cls_tp = tp[y == cls]
            x_i = features_2d[y == cls]
            x_tp = x_i[cls_tp]
            x_fp = x_i[~cls_tp]
            # if self.parent.ui.current_classificator_comboBox.currentText() == 'XGBoost':
            #     cls = self.get_old_class_label(cls)
            style = context.group_table.table_widget.model().cell_data_by_idx_col_name(
                cls, 'Style')
            color = 'orange' if style is None else style['color'].name()
            plot_widget.canvas.gca().scatter(x_tp[:, 0], x_tp[:, 1], marker=markers[cls],
                                             color=color, edgecolor='black', s=60)
            plot_widget.canvas.gca().scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=60, color=color,
                                             edgecolor='black')
        plot_widget.canvas.gca().set_xlabel(dr_method + f'-1 ({evr[0] * 100:.2f} %)' if evr[0] != .0
                                            else dr_method + '-1',
                                            fontsize=int(environ['axis_label_font_size']))
        plot_widget.canvas.gca().set_ylabel(dr_method + f'-2 ({evr[1] * 100:.2f} %)' if evr[1] != .0
                                            else dr_method + '-2',
                                            fontsize=int(environ['axis_label_font_size']))
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plt.close()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_dm_plot(self, cl_type: str) -> None:
        """
        Update the confusion matrix plot for the given classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to update the confusion matrix plot.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.dm_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()

        model = self.data[cl_type]['model']
        x_test = self.data[cl_type]['x_test']
        y_test = self.data[cl_type]['y_test']
        target_names = self.data[cl_type]['target_names']
        font_size = int(environ['axis_label_font_size'])

        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, colorbar=False)
        try:
            ax.xaxis.set_ticklabels(target_names, fontsize=font_size)
            ax.yaxis.set_ticklabels(target_names, fontsize=font_size)
        except ValueError as err:
            error(err)
        ax.set_title(f"Confusion Matrix for\n{model.__class__.__name__}\non the test data")
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_roc_plot(self, cl_type: str) -> None:
        """
        Update the ROC curve plot for the given classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to update the ROC curve plot.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.roc_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        y_score = self.data[cl_type]['y_score_test']
        y_onehot_test = self.data[cl_type]['y_onehot_test']
        # store the fpr, tpr, and roc_auc for all averaging strategies
        fpr, tpr, roc_auc = {}, {}, {}
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        n_classes = len(self.data[cl_type]['target_names'])
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
        # Average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
                color="deeppink", linestyle=":", linewidth=3, )
        ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC = {roc_auc['macro']:.2f})",
                color="turquoise", linestyle=":", linewidth=3, )
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)",
                color=environ['primaryColor'])
        for class_id, color in zip(range(n_classes),
                                   self._get_plot_colors(self.data[cl_type]['model'].classes_)):
            RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_score[:, class_id],
                                             name=self.data[cl_type]['target_names'][class_id],
                                             color=color, ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_ylabel("True Positive Rate", fontsize=int(environ['axis_label_font_size']))
        ax.set_title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_pr_plot(self, cl_type: str) -> None:
        """
        Update the precision-recall curve plot for the given classifier type.

        Parameters
        ----------
        cl_type : str
            The type of classifier for which to update the precision-recall curve plot.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.pr_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        classes = self.data[cl_type]['model'].classes_
        target_names = self.data[cl_type]['target_names']
        y_score_dec_func = self.data[cl_type]['y_score_decision']
        y_test_bin = label_binarize(self.data[cl_type]['y_test'],
                                    classes=list(set(self.data[cl_type]['y']))) \
            if len(target_names) > 2 else None
        precision, recall, average_precision = {}, {}, {}
        n_classes = len(classes)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i],
                                                                y_score_dec_func[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i],
                                                           y_score_dec_func[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(),
                                                                        y_score_dec_func.ravel())
        average_precision["micro"] = average_precision_score(y_test_bin, y_score_dec_func,
                                                             average="micro")
        display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                         average_precision=average_precision["micro"])
        display.plot(ax=ax, name="Micro-average precision-recall", color="deeppink", linestyle=":",
                     linewidth=3)
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="deeppink", alpha=0.2)
            ax.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02), color="deeppink")
        for i, color in zip(range(n_classes), self._get_plot_colors(classes)):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i], )
            target_name = target_names[i] if i < len(target_names) else 'NoName'
            display.plot(ax=ax, name=f"Precision-recall for {target_name}", color=color,
                         linewidth=2)
        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best",
                  prop={'size': int(environ['plot_font_size'])})
        ax.set_title("Extension of Precision-Recall curve to multi-class")
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_roc_plot_bin(self, cl_type: str) -> None:
        """
        Update the ROC plot for binary classification.

        Clears the current plot and generates a new ROC curve plot based on the model and data
        for the specified classification type (`cl_type`). The plot includes a chance level line
        and the ROC curve of the model.

        Parameters
        ----------
        cl_type : str
            The type of classifier ('binary' in this case) used to fetch the model and data
            for plotting.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.roc_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        x_test = self.data[cl_type]['x_test']
        y_test = self.data[cl_type]['y_test']
        target_names = self.data[cl_type]['target_names']
        font_size = int(environ['axis_label_font_size'])
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)",
                color=environ['primaryColor'])
        RocCurveDisplay.from_estimator(model, x_test, y_test, name=target_names[0],
                                       color='darkorange', ax=ax)
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=font_size)
        ax.set_ylabel("True Positive Rate", fontsize=font_size)
        ax.set_title("ROC curve")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_pr_plot_bin(self, cl_type: str) -> None:
        """
        Update the Precision-Recall plot for binary classification.

        Clears the current plot and generates a new Precision-Recall curve plot based on
        the model and data provided for the specified classification type (`cl_type`).
        The plot includes the Precision-Recall curve of the model.

        Parameters
        ----------
        cl_type : str
            The type of classifier ('binary' in this case) used to fetch the model and data
            for plotting.

        Returns
        -------
        None
        """
        plot_widget = get_parent(self.parent, "MainWindow").ui.pr_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        y_score_decision = self.data[cl_type]['y_score_decision']
        y_test = self.data[cl_type]['y_test']
        pos_label = self.data[cl_type]['model'].classes_[0]
        if len(y_score_decision.shape) > 1 and y_score_decision.shape[1] > 1:
            y_score_decision = y_score_decision[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_score_decision, name=cl_type, ax=ax,
                                                color='darkorange', pos_label=pos_label)
        ax.set_title("2-class Precision-Recall curve")
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size'])})
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _build_permutation_importance_plot(self, cl_type, test: bool = False) -> None:
        """
        Build and update the permutation importance plot.

        Clears the current plot and generates a permutation importance plot for the model
        based on the specified classification type (`cl_type`). The plot shows the importance
        of features as determined by permutation importance.

        Parameters
        ----------
        cl_type : str
            The type of classifier used to fetch the model and data for plotting.
        test : bool, optional
            Whether to use the test set (True) or the training set (False). Default is False.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.perm_imp_test_plot_widget if test else mw.ui.perm_imp_train_plot_widget
        plot_widget.canvas.gca().cla()
        ax = plot_widget.canvas.gca()
        model = self.data[cl_type]['model']
        x = self.data[cl_type]['x_test'] if test else self.data[cl_type]['x_train']
        y = self.data[cl_type]['y_test'] if test else self.data[cl_type]['y_train']
        binary = len(model.classes_) == 2
        scoring = 'f1' if binary else 'f1_micro'
        with parallel_backend('multiprocessing', n_jobs=-1):
            result = permutation_importance(model, x, y, scoring=scoring,
                                            n_jobs=-1,
                                            random_state=mw.ui.random_state_sb.value())
        sorted_importance_idx = result.importances_mean.argsort()
        importance = pd.DataFrame(
            result.importances[sorted_importance_idx].T,
            columns=x.columns[sorted_importance_idx],
        )
        importance.plot.box(vert=False, whis=10, ax=ax)
        ax.set_title(f"Permutation Importances ({'test' if test else 'train'} set)")
        try:
            ax.axvline(x=0, color="k", linestyle="--")
        except np.linalg.LinAlgError:
            pass
        ax.set_xlabel("Decrease in accuracy score")
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_plot_tree(self, cl_type: str) -> None:
        """
        Update the decision tree or random forest tree plot.

        Clears the current plot and generates a tree plot based on the model and data for
        the specified classification type (`cl_type`). For Random Forest, the specific tree
        selected by the user will be plotted.

        Parameters
        ----------
        cl_type : str
            The type of classifier ('Decision Tree' or 'Random Forest') used to fetch the
            model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.tree_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        if cl_type not in ['Decision Tree', 'Random Forest']:
            plot_widget.canvas.gca().text(.5, .5, 'No data', color='red')
            plot_widget.canvas.draw()
            return
        model = self.data[cl_type]['model']
        if cl_type == 'Random Forest':
            model = model.estimators_[mw.ui.current_tree_spinBox.value()]
        feature_names = self.data[cl_type]['feature_names']
        class_names = self.data[cl_type]['target_names']
        plot_tree(model, feature_names=feature_names, fontsize=int(environ['plot_font_size']),
                  class_names=class_names, ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def build_partial_dependence_plot(self, cl_type: str) -> None:
        """
        Build and update the partial dependence plot.

        Clears the current plot and generates a partial dependence plot for the model based on
        the specified classification type (`cl_type`). The plot shows the effect of specified
        features on the predicted outcome.

        Parameters
        ----------
        cl_type : str
            The type of classifier used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.partial_depend_plot_widget
        plot_widget.canvas.figure.clf()
        ax = plot_widget.canvas.gca()
        ax.cla()
        x = self.data[cl_type]['x']
        model = self.data[cl_type]['model']
        f1 = mw.ui.current_dep_feature1_comboBox.currentText()
        f2 = mw.ui.current_dep_feature2_comboBox.currentText()
        with parallel_backend('multiprocessing'):
            PartialDependenceDisplay.from_estimator(model, x, [f1, (f1, f2)],
                                                    target=model.classes_[0],
                                                    feature_names=list(x.columns), ax=ax)
        plot_widget.canvas.figure.tight_layout()
        plot_widget.canvas.draw()
        plot_widget.canvas.figure.tight_layout()

    def _update_features_plot(self, cl_type: str) -> None:
        """
        Update the feature importance plot for linear models.

        Clears the current plot and generates a bar plot showing the average effect of features
        on the prediction for a linear model. Features are weighted by their frequency of
        appearance.

        Parameters
        ----------
        cl_type : str
            The type of classifier ('linear' model) used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.features_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        target_names = self.data[cl_type]['target_names']
        average_feature_effects = model.coef_ * np.asarray(
            self.data[cl_type]['x_train'].mean(axis=0)).ravel()
        top_indices = []
        for i, label in enumerate(target_names):
            if i >= len(average_feature_effects):
                break
            top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
            top_indices = top5 if i == 0 else np.concatenate((top_indices, top5))
        top_indices = np.unique(top_indices)
        bar_size, padding = 0.5, 0.75
        y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)

        for i, label in enumerate(target_names):
            if i >= average_feature_effects.shape[0]:
                break
            ax.barh(y_locs + (i - 2) * bar_size, average_feature_effects[i, top_indices],
                    height=bar_size, label=label,
                    color=self._get_plot_colors(model.classes_)[i])
        ax.set(yticks=y_locs, yticklabels=self.data[cl_type]['feature_names'][top_indices],
               ylim=[0 - 4 * bar_size, len(top_indices) * (4 * bar_size + padding) - 4 * bar_size])
        ax.legend(loc="best")
        ax.set_xlabel("Average feature effect on the original data",
                      fontsize=int(environ['axis_label_font_size']))
        ax.set_ylabel("Feature", fontsize=int(environ['axis_label_font_size']))
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_features_plot_random_forest(self, cl_type: str) -> None:
        """
        Update the feature importance plot for Random Forest.

        Clears the current plot and generates a bar plot showing feature importance as
        determined by the Random Forest model. The plot includes the Mean Decrease in
        Impurity (MDI) importance.

        Parameters
        ----------
        cl_type : str
            The type of classifier ('Random Forest') used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.features_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        title = f"{cl_type} Feature Importance (MDI)"
        if mw.ui.dataset_type_cb.currentText() == 'Decomposed':
            mdi = pd.Series(model.feature_importances_,
                            index=model.feature_names_in_).sort_values(ascending=True)
            mdi = mdi[mdi > 0]
            ax.barh(mdi.index, mdi, color=environ['primaryColor'])
            ax.set_xlabel('MDI importance', fontsize=int(environ['axis_label_font_size']))
            ax.set_ylabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=int(environ['axis_label_font_size']))
        else:
            k = [float(re.sub(r'k|_a|_x0', '', i)) for i in model.feature_names_in_]
            ser = pd.Series(model.feature_importances_, index=k)
            ax.plot(ser)
            ax.set_xlabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=int(environ['axis_label_font_size']))
            ax.set_ylabel('MDI importance', fontsize=int(environ['axis_label_font_size']))
        ax.set_title(title)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_xgboost_tree_plot(self, idx: int = 0) -> None:
        """
        Update the XGBoost tree plot.

        Clears the current plot and generates a visualization of a specified tree from the
        XGBoost model. The tree index can be specified to visualize different trees.

        Parameters
        ----------
        idx : int, optional
            The index of the tree to plot. Default is 0.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.tree_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data['XGBoost']['model']
        try:
            xgboost.plot_tree(model, ax=ax, num_trees=idx,
                              yes_color=environ['secondaryDarkColor'],
                              no_color=environ['primaryColor'])
        except ImportError as e:
            mw.progress.show_error(e)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_features_plot_xgboost(self) -> None:
        """
        Update the feature importance plot for XGBoost.

        Clears the current plot and generates a plot showing feature importance for the
        XGBoost model. The plot includes importance scores for the top features.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.features_plot_widget
        model = self.data['XGBoost']['model']
        ax = plot_widget.canvas.gca()
        ax.cla()
        max_num_features = mw.ui.feature_display_max_spinBox.value() \
            if mw.ui.feature_display_max_checkBox.isChecked() else None
        try:
            xgboost.plot_importance(model, ax, max_num_features=max_num_features, grid=False)
        except ValueError as e:
            error(e)
            return
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _build_calibration_plot(self, cl_type: str) -> None:
        """
        Build and update the calibration plot.

        Clears the current plot and generates a calibration plot based on the model and data
        for the specified classification type (`cl_type`). The plot shows the calibration of
        the model's probabilities.

        Parameters
        ----------
        cl_type : str
            The type of classifier used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.calibration_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        x_test = self.data[cl_type]['x_test']
        y_test = self.data[cl_type]['y_test']
        with parallel_backend('multiprocessing'):
            CalibrationDisplay.from_estimator(model, x_test, y_test, pos_label=model.classes_[0],
                                              ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _build_dev_curve_plot(self, cl_type: str) -> None:
        """
        Build and update the detection error tradeoff (DET) curve plot.

        Clears the current plot and generates a DET curve based on the model and data for
        the specified classification type (`cl_type`). The plot shows the tradeoff between
        false positive rate and false negative rate.

        Parameters
        ----------
        cl_type : str
            The type of classifier used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.det_curve_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        x_test = self.data[cl_type]['x_test']
        y_test = self.data[cl_type]['y_test']
        with parallel_backend('multiprocessing'):
            DetCurveDisplay.from_estimator(model, x_test, y_test, pos_label=model.classes_[0],
                                           ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_pca_plots(self) -> None:
        """
        Update all PCA plots and fields.

        Updates the PCA score plots, PCA loadings plots, and PCA features table view based
        on the results from the PCA model.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        model_results = self.data['PCA']
        y_train_test = model_results['y_train_test']
        features_2d = model_results['features_2d']
        evr = model_results['explained_variance_ratio']

        self._update_scores_plot_pca(features_2d, y_train_test,
                                     mw.ui.pca_scores_plot_widget,
                                     evr, model_results['target_names'])
        if 'loadings' in model_results:
            mw.ui.pca_features_table_view.model().set_dataframe(model_results['loadings'])
            self._update_pca_loadings_plot(model_results['loadings'], evr)

    def _update_scores_plot_pca(self, features_in_2d: np.ndarray, y: list[int], plot_widget,
                                evr, target_names) -> None:
        """
        Update the PCA scores plot.

        Clears the current plot and generates a PCA score plot showing the 2D projection of
        the data. Different classes are represented with different markers and colors.

        Parameters
        ----------
        features_in_2d : np.ndarray
            The 2D transformed features for the PCA plot.
        y : list[int]
            The true labels for the data points.
        plot_widget : object
            The widget where the PCA scores plot will be displayed.
        evr : np.ndarray
            The explained variance ratio for the principal components.
        target_names : list[str]
            The names of the target classes.

        Returns
        -------
        None
        """
        context = get_parent(self.parent, "Context")
        ax = plot_widget.canvas.gca()
        ax.cla()
        classes = np.unique(y)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P',
                   '*']
        for i, cls in enumerate(classes):
            x_i = features_in_2d[y == cls]
            mrkr = markers[i]
            color = context.group_table.table_widget.model().cell_data_by_idx_col_name(
                cls, 'Style')['color'].name()
            ax.scatter(x_i[:, 0], x_i[:, 1], marker=mrkr, color=color,
                       edgecolor='black', s=60, label=target_names[i])
        ax.set_xlabel(f'PC-1 ({evr[0] * 100:.2f}%)', fontsize=int(environ['plot_font_size']))
        ax.set_ylabel(f'PC-2 ({evr[1] * 100:.2f}%)', fontsize=int(environ['plot_font_size']))
        ax.legend(loc="best", prop={'size': int(environ['plot_font_size']) - 2})
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def _update_pca_loadings_plot(self, loadings, evr: float) -> None:
        """
        Update the PCA loadings plot.

        Clears the current plot and generates a PCA loadings plot showing the loadings of
        features on the first two principal components.

        Parameters
        ----------
        loadings : pd.DataFrame
            The PCA loadings for the features.
        evr : np.ndarray
            The explained variance ratio for the principal components.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.pca_loadings_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        features_names = list(loadings.axes[0])
        ax.scatter(loadings['PC1'], loadings['PC2'], color=environ['primaryColor'], s=60)
        for i, txt in enumerate(features_names):
            ax.annotate(txt, (loadings['PC1'][i], loadings['PC2'][i]))
        ax.set_xlabel(f'PC-1 ({evr[0] * 100:.2f} %)', fontsize=int(environ['plot_font_size']))
        ax.set_ylabel(f'PC-2 ({evr[1] * 100:.2f} %)', fontsize=int(environ['plot_font_size']))
        ax.set_title("PCA Loadings")
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass

    def update_optuna_plots(self, cl_type: str | None = None):
        """
        Update Optuna plots for hyperparameter optimization.

        Updates the plots for optimization history, parameter importance, and contour plots
        based on the results from the Optuna study. Different plots are generated depending
        on the type of classifier.

        Parameters
        ----------
        cl_type : str, optional
            The type of classifier to fetch the Optuna study for. If None, uses the current
            classifier selected in the UI.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        if cl_type is None:
            cl_type = mw.ui.current_classificator_comboBox.currentText()
        study = get_study(cl_type)
        if len(study.trials) == 0:
            return
        params = get_optuna_params()[cl_type]
        template = 'plotly_white' if mw.ui.sun_Btn.isChecked() else 'plotly_dark'
        for p, plot_widget in zip(params, (mw.ui.cont_1, mw.ui.cont_3, mw.ui.cont_2, mw.ui.cont_4,
                                           mw.ui.cont_5, mw.ui.cont_6)):
            try:
                fig = optuna.visualization.plot_contour(study, params=p, target_name="ROC AUC")
            except ValueError as e:
                error(e)
                break
            fig.update_layout(template=template)
            plot_widget.setHtml(fig.to_html(include_plotlyjs='cdn'))
            plot_widget.page().setBackgroundColor(mw.attrs.plot_background_color)
        if cl_type == 'XGBoost':
            fig = optuna.visualization.plot_contour(get_study('XGBoost_1'),
                                                    params=['n_estimators', 'learning_rate'],
                                                    target_name="ROC AUC")
            fig.update_layout(template=template)
            mw.ui.cont_6.setHtml(fig.to_html(include_plotlyjs='cdn'))
            mw.ui.cont_6.page().setBackgroundColor(mw.attrs.plot_background_color)
        fig = optuna.visualization.plot_optimization_history(study, target_name="ROC AUC")
        fig.update_layout(template=template)
        mw.ui.hist_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        mw.ui.hist_plot.page().setBackgroundColor(mw.attrs.plot_background_color)
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(template=template)
        mw.ui.param_imp_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        mw.ui.param_imp_plot.page().setBackgroundColor(mw.attrs.plot_background_color)

    def build_learning_curve(self, cl_type: str) -> None:
        """
        Build and update the learning curve plot.

        Clears the current plot and generates a learning curve plot showing the performance
        of the model as a function of training size.

        Parameters
        ----------
        cl_type : str
            The type of classifier used to fetch the model and data for plotting.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_widget = mw.ui.learning_plot_widget
        ax = plot_widget.canvas.gca()
        ax.cla()
        model = self.data[cl_type]['model']
        x = self.data[cl_type]['x']
        y = self.data[cl_type]['y']
        with parallel_backend('multiprocessing', n_jobs=-1):
            LearningCurveDisplay.from_estimator(model, x, y, ax=ax)
        try:
            plot_widget.canvas.figure.tight_layout()
            plot_widget.canvas.draw()
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
