import os
import re
from asyncio import gather
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from logging import error, warning
from os import environ


import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost
from asyncqtpy import asyncSlot
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame, Series
from pyqtgraph import BarGraphItem, InfiniteLine
from qtpy.QtGui import QColor
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc, precision_recall_curve, \
    average_precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree
from imblearn.over_sampling import RandomOverSampler

from modules.default_values import classificator_funcs
from modules.static_functions import insert_table_to_text_edit
from modules.undo_redo import CommandAfterFittingStat


def update_plot_tree(clf, plot_widget, feature_names, class_names) -> None:
    ax = plot_widget.canvas.axes
    ax.cla()
    plot_tree(clf, feature_names=feature_names, class_names=class_names, ax=ax)
    try:
        plot_widget.canvas.draw()
    except ValueError:
        pass
    plot_widget.canvas.figure.tight_layout()


class StatAnalysisLogic:

    def __init__(self, parent):
        self.parent = parent
        self.classificator_funcs = classificator_funcs()
        self.latest_stat_result = {}
        self.top_features = {}
        self.old_labels = None
        self.new_labels = None
        self.params = {'score_plots': {'QDA': self.parent.ui.qda_scores_plot_widget,
                                       'Logistic regression': self.parent.ui.lr_scores_plot_widget,
                                       'NuSVC': self.parent.ui.svc_scores_plot_widget,
                                       'Nearest Neighbors': self.parent.ui.nearest_scores_plot_widget,
                                       'GPC': self.parent.ui.gpc_scores_plot_widget,
                                       'Decision Tree': self.parent.ui.dt_scores_plot_widget,
                                       'Naive Bayes': self.parent.ui.nb_scores_plot_widget,
                                       'Random Forest': self.parent.ui.rf_scores_plot_widget,
                                       'AdaBoost': self.parent.ui.ab_scores_plot_widget,
                                       'MLP': self.parent.ui.mlp_scores_plot_widget,
                                       'XGBoost': self.parent.ui.xgboost_scores_plot_widget},
                       'feature_plots': {'LDA': self.parent.ui.lda_features_plot_widget,
                                         'Logistic regression': self.parent.ui.lr_features_plot_widget,
                                         'NuSVC': self.parent.ui.svc_features_plot_widget},
                       'dm_plots': {'LDA': self.parent.ui.lda_dm_plot, 'QDA': self.parent.ui.qda_dm_plot,
                                    'Logistic regression': self.parent.ui.lr_dm_plot,
                                    'NuSVC': self.parent.ui.svc_dm_plot,
                                    'Nearest Neighbors': self.parent.ui.nearest_dm_plot,
                                    'GPC': self.parent.ui.gpc_dm_plot,
                                    'Decision Tree': self.parent.ui.dt_dm_plot,
                                    'Naive Bayes': self.parent.ui.nb_dm_plot,
                                    'Random Forest': self.parent.ui.rf_dm_plot,
                                    'AdaBoost': self.parent.ui.ab_dm_plot,
                                    'MLP': self.parent.ui.mlp_dm_plot,
                                    'XGBoost': self.parent.ui.xgboost_dm_plot},
                       'pr_plots': {'LDA': self.parent.ui.lda_pr_plot, 'QDA': self.parent.ui.qda_pr_plot,
                                    'Logistic regression': self.parent.ui.lr_pr_plot,
                                    'NuSVC': self.parent.ui.svc_pr_plot,
                                    'Nearest Neighbors': self.parent.ui.nearest_pr_plot,
                                    'GPC': self.parent.ui.gpc_pr_plot,
                                    'Decision Tree': self.parent.ui.dt_pr_plot,
                                    'Naive Bayes': self.parent.ui.nb_pr_plot,
                                    'Random Forest': self.parent.ui.rf_pr_plot,
                                    'AdaBoost': self.parent.ui.ab_pr_plot,
                                    'MLP': self.parent.ui.mlp_pr_plot,
                                    'XGBoost': self.parent.ui.xgboost_pr_plot},
                       'roc_plots': {'LDA': self.parent.ui.lda_roc_plot, 'QDA': self.parent.ui.qda_roc_plot,
                                     'Logistic regression': self.parent.ui.lr_roc_plot,
                                     'NuSVC': self.parent.ui.svc_roc_plot,
                                     'Nearest Neighbors': self.parent.ui.nearest_roc_plot,
                                     'GPC': self.parent.ui.gpc_roc_plot,
                                     'Decision Tree': self.parent.ui.dt_roc_plot,
                                     'Naive Bayes': self.parent.ui.nb_roc_plot,
                                     'Random Forest': self.parent.ui.rf_roc_plot,
                                     'AdaBoost': self.parent.ui.ab_roc_plot,
                                     'MLP': self.parent.ui.mlp_roc_plot,
                                     'XGBoost': self.parent.ui.xgboost_roc_plot},
                       }

    @asyncSlot()
    async def do_fit_classificator(self, cl_type: str) -> None:
        """
        Обучение выбранного классификатора, обновление всех графиков.
        @param cl_type: название метода классификации
        @return: None
        """
        main_window = self.parent
        main_window.time_start = datetime.now()
        main_window.ui.statusBar.showMessage('Fitting model...')
        main_window.close_progress_bar()
        main_window.open_progress_bar(max_value=0)
        main_window.open_progress_dialog("Fitting Classificator...", "Cancel", maximum=0)
        X, Y, feature_names, target_names, _ = self.dataset_for_ml()

        if cl_type == 'XGBoost':
            Y = self.corrected_class_labels(Y)
        y_test_bin = None
        test_size = main_window.ui.test_data_ratio_spinBox.value() / 100.
        if len(target_names) > 2:
            Y_bin = label_binarize(Y, classes=list(np.unique(Y)))
            _, _, _, y_test_bin = train_test_split(X, Y_bin, test_size=test_size)
        if main_window.ui.random_state_cb.isChecked():
            rng = np.random.RandomState(main_window.ui.random_state_sb.value())
        else:
            rng = None
        ros = RandomOverSampler(random_state=rng)
        X, Y = ros.fit_resample(X, Y)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=rng)
        executor = ProcessPoolExecutor()
        func = self.classificator_funcs[cl_type]
        # Для БОльших датасетов возможно лучше будет ProcessPoolExecutor. Но таких пока нет
        main_window.current_executor = executor
        params = None
        cl_types_using_pca_plsda = list(self.classificator_funcs.keys())
        cl_types_using_pca_plsda.remove('LDA')
        cl_types_using_pca_plsda.remove('PCA')
        cl_types_using_pca_plsda.remove('PLS-DA')
        use_pca = self.parent.ui.use_pca_checkBox.isChecked()
        if cl_type == 'MLP' and main_window.ui.groupBox_mlp.isChecked():
            params = {'activation': main_window.ui.activation_comboBox.currentText(),
                      'solver': main_window.ui.solver_mlp_combo_box.currentText(),
                      'hidden_layer_sizes': main_window.ui.mlp_layer_size_spinBox.value(),
                      'use_pca': use_pca}
        elif cl_type in cl_types_using_pca_plsda:
            params = {'use_pca': use_pca}

        with executor:
            if params is not None:
                main_window.current_futures = [
                    main_window.loop.run_in_executor(executor, func, x_train, y_train, x_test, y_test, params)]
            else:
                main_window.current_futures = [main_window.loop.run_in_executor(executor, func, x_train, y_train,
                                                                                x_test, y_test)]
            for future in main_window.current_futures:
                future.add_done_callback(main_window.progress_indicator)
            result = await gather(*main_window.current_futures)
        if main_window.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            main_window.close_progress_bar()
            main_window.ui.statusBar.showMessage('Fitting cancelled.')
            return
        result = result[0]
        result['X'] = X
        result['y_train'] = y_train
        result['use_pca'] = use_pca
        result['y_test'] = y_test
        result['x_train'] = x_train
        result['x_test'] = x_test
        result['target_names'] = target_names
        result['feature_names'] = feature_names
        result['y_test_bin'] = y_test_bin
        command = CommandAfterFittingStat(main_window, result, cl_type, "Fit model %s" % cl_type)
        main_window.undoStack.push(command)
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Fitting completed', 10000)

    def corrected_class_labels(self, Y: list[int]) -> list[int]:
        uniq = np.unique(Y)
        n_classes = len(uniq)
        self.old_labels = uniq
        self.new_labels = list(np.arange(0, n_classes, 1))
        new_Y = []
        for i in Y:
            idx = np.argwhere(uniq == i)[0][0]
            new_Y.append(self.new_labels[idx])
        return new_Y

    def get_old_class_label(self, i: int) -> int:
        idx = np.argwhere(self.new_labels == i)[0][0]
        return self.old_labels[idx]

    def dataset_for_ml(self) -> tuple[DataFrame, list[int], list[str], np.ndarray, DataFrame] | None:
        """
        Выбор данных для обучения из датасета
        @return:
        X: DataFrame. Columns - fealures, rows - samples
        Y: list[int]. True labels
        feature_names: feature names (lol)
        target_names: classes names
        """
        main_window = self.parent
        selected_dataset = main_window.ui.dataset_type_cb.currentText()
        if selected_dataset == 'Smoothed':
            model = main_window.ui.smoothed_dataset_table_view.model()
        elif selected_dataset == 'Baseline corrected':
            model = main_window.ui.baselined_dataset_table_view.model()
        elif selected_dataset == 'Deconvoluted':
            model = main_window.ui.deconvoluted_dataset_table_view.model()
        else:
            return
        q_res = model.dataframe()
        if main_window.ui.classes_lineEdit.text() != '':
            v = list(main_window.ui.classes_lineEdit.text().strip().split(','))
            classes = []
            for i in v:
                classes.append(int(i))
            if len(classes) > 1:
                q_res = model.query_result_with_list('Class == @input_list', classes)
        y = list(q_res['Class'])
        classes = np.unique(q_res['Class'].values)
        if main_window.predict_logic.is_production_project:
            target_names = None
        else:
            target_names = main_window.ui.GroupsTable.model().dataframe().loc[classes]['Group name'].values

        return q_res.iloc[:, 2:], y, list(q_res.axes[1][2:]), target_names, q_res.iloc[:, 1]

    def _get_plot_colors(self, classes: list[int]) -> list[str]:
        plt_colors = []
        for cls in classes:
            clr = self.parent.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        return plt_colors

    def get_current_dataset_type_cb(self):
        ct = self.parent.ui.dataset_type_cb.currentText()
        if ct == 'Smoothed':
            return self.parent.ui.smoothed_dataset_table_view.model()
        elif ct == 'Baseline corrected':
            return self.parent.ui.baselined_dataset_table_view.model()
        elif ct == 'Deconvoluted':
            return self.parent.ui.deconvoluted_dataset_table_view.model()
        else:
            return None

    def update_stat_report_text(self):
        idx = self.parent.ui.stat_tab_widget.currentIndex()
        if idx > 14 or idx < 0:
            self.parent.ui.stat_report_text_edit.setText('')
            return
        classificator_type = list(self.classificator_funcs.keys())[idx]
        if classificator_type not in self.latest_stat_result or 'metrics_result' \
                not in self.latest_stat_result[classificator_type]:
            self.parent.ui.stat_report_text_edit.setText('')
            return
        if classificator_type in self.top_features:
            top = self.top_features[classificator_type]
        else:
            top = None
        model_results = self.latest_stat_result[classificator_type]
        self.update_report_text(model_results['metrics_result'], model_results['cv_scores'], top,
                                model_results['model'], classificator_type)

    def update_report_text(self, metrics_result: dict, cv_scores=None, top=None, model=None, classificator_type=None) \
            -> None:
        """
        Set report text
        @param model:
        @param cv_scores:
        @param metrics_result:
        @param top: top 5 features for each class
        @return:
        """
        text = '\n' + 'Accuracy score (test data): {!s}%'.format(metrics_result['accuracy_score']) + '\n' \
               + 'Accuracy score (train data): {!s}%'.format(metrics_result['accuracy_score_train']) + '\n' \
               + 'Precision score: {!s}%'.format(metrics_result['precision_score']) + '\n' \
               + 'Recall score: {!s}%'.format(metrics_result['recall_score']) + '\n' \
               + 'F1 score: {!s}%'.format(metrics_result['f1_score']) + '\n' \
               + 'F_beta score: {!s}%'.format(metrics_result['fbeta_score']) + '\n' \
               + 'Hamming loss score: {!s}%'.format(metrics_result['hamming_loss']) + '\n' \
               + 'Jaccard score: {!s}%'.format(metrics_result['jaccard_score']) + '\n' + '\n'
        if top is not None:
            text += 'top 5 features per class:' + '\n' + str(top) + '\n'
        if cv_scores is not None:
            text += '\n' + "Cross validated %0.2f accuracy with a standard deviation of %0.2f" \
                    % (cv_scores.mean(), cv_scores.std()) + '\n'
        if model is not None and isinstance(model, GridSearchCV):
            text += '\n' + 'Mean Accuracy of best estimator: %.3f' % model.best_score_ + '\n'
            text += 'Config: %s' % model.best_params_ + '\n'
        if classificator_type == 'Random Forest':
            text += 'N Trees: %s' % len(model.best_estimator_.estimators_) + '\n'
        elif classificator_type == 'AdaBoost':
            text += 'N Trees: %s' % len(model.estimators_) + '\n'
        self.parent.ui.stat_report_text_edit.setText(text)
        if metrics_result['classification_report'] is None:
            return
        headers = [' ']
        rows = []
        for i in metrics_result['classification_report'].split('\n')[0].strip().split(' '):
            if i != '':
                headers.append(i)
        for i, r in enumerate(metrics_result['classification_report'].split('\n')):
            new_row = []
            if r == '' or i == 0:
                continue
            rr = r.split('  ')
            for c in rr:
                if c == '':
                    continue
                new_row.append(c)
            if new_row[0].strip() == 'accuracy':
                new_row = [new_row[0], '', '', new_row[1], new_row[2]]
            rows.append(new_row)

        insert_table_to_text_edit(self.parent.ui.stat_report_text_edit.textCursor(), headers, rows)

    # region plots update
    def update_scores_plot(self, features_in_2d: np.ndarray, y: list[int], y_pred: list[int], model,
                           model_2d, plot_widget, explained_variance_ratio, use_pca: bool) -> None:
        """
        @param plot_widget:
        @param features_in_2d: transformed 2d
        @param y: true labels
        @param y_pred: predicted labels
        @param model: classificator
        @param model_2d: 2d model classificator
        @param explained_variance_ratio:
        @return:
        """
        plot_widget.canvas.axes.cla()
        if explained_variance_ratio is None:
            explained_variance_ratio = [0, 0]
        classes = model.classes_
        plt_colors = []
        for cls in classes:
            clr = self.parent.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter(140).name())
        tp = y == y_pred  # True Positive
        cmap = LinearSegmentedColormap('', None).from_list('', plt_colors)
        DecisionBoundaryDisplay.from_estimator(model_2d, features_in_2d, grid_resolution=1000, eps=.5,
                                               antialiased=True, cmap=cmap, ax=plot_widget.canvas.axes)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            true_positive_of_class = tp[y == cls]
            x_i = features_in_2d[y == cls]
            x_tp = x_i[true_positive_of_class]
            x_fp = x_i[~true_positive_of_class]
            mrkr = markers[cls]
            if plot_widget == self.parent.ui.xgboost_scores_plot_widget:
                cls = self.get_old_class_label(cls)
            color = self.parent.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            plot_widget.canvas.axes.scatter(x_tp[:, 0], x_tp[:, 1], marker=mrkr, color=color,
                                            edgecolor='black', s=60)
            plot_widget.canvas.axes.scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=60, color=color)
        dr_method = 'PC' if use_pca else 'PLS-DA'
        plot_widget.canvas.axes.set_xlabel(dr_method + '-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                                           fontsize=self.parent.axis_label_font_size)
        plot_widget.canvas.axes.set_ylabel(dr_method + '-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                                           fontsize=self.parent.axis_label_font_size)
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()
        plt.close()

    def update_features_plot(self, X_train: DataFrame, model, feature_names: list[str], target_names: list[str],
                             plt_colors, plot_widget) -> DataFrame:
        """
        Update features plot

        @param plot_widget:
        @param plt_colors:
        @param X_train: dataframe with train data
        @param model: model
        @param feature_names: name for features
        @param target_names: classes name
        @return: top features per class
        """

        # learned coefficients weighted by frequency of appearance
        feature_names = np.array(feature_names)
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        average_feature_effects = model.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
        for i, label in enumerate(target_names):
            if i >= len(average_feature_effects):
                break
            top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
            if i == 0:
                top = DataFrame(feature_names[top5], columns=[label])
                top_indices = top5
            else:
                top[label] = feature_names[top5]
                top_indices = np.concatenate((top_indices, top5), axis=None)
        top_indices = np.unique(top_indices)
        predictive_words = feature_names[top_indices]
        # plot feature effects
        bar_size = 0.25
        padding = 0.75
        y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)
        ax = plot_widget.canvas.axes
        ax.cla()
        for i, label in enumerate(target_names):
            if i >= average_feature_effects.shape[0]:
                break
            ax.barh(y_locs + (i - 2) * bar_size, average_feature_effects[i, top_indices], height=bar_size, label=label,
                    color=plt_colors[i])
        ax.set(yticks=y_locs, yticklabels=predictive_words,
               ylim=[0 - 4 * bar_size, len(top_indices) * (4 * bar_size + padding) - 4 * bar_size])
        ax.legend(loc="best")
        ax.set_xlabel("Average feature effect on the original data", fontsize=self.parent.axis_label_font_size)
        ax.set_ylabel("Feature", fontsize=self.parent.axis_label_font_size)
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()
        return top

    def update_roc_plot_bin(self, model, x_test, y_test, target_names, plot_widget) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)", color=self.parent.theme_colors['primaryColor'])
        RocCurveDisplay.from_estimator(model, x_test, y_test,
                                       name=target_names[1], color='darkorange', ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=self.parent.axis_label_font_size)
        ax.set_ylabel("True Positive Rate", fontsize=self.parent.axis_label_font_size)
        ax.set_title("Receiver Operating Characteristic (ROC) curve")
        ax.legend(loc="best", prop={'size': self.parent.plot_font_size})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_roc_plot(self, y_score, y_onehot_test, target_names, plt_colors, plot_widget) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        # store the fpr, tpr, and roc_auc for all averaging strategies
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        n_classes = len(target_names)
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
        ax.plot([0, 1], [0, 1], "--", label="chance level (AUC = 0.5)", color=self.parent.theme_colors['primaryColor'])
        for class_id, color in zip(range(n_classes), plt_colors):
            RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_score[:, class_id],
                                             name=target_names[class_id], color=color, ax=ax, )
        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=self.parent.axis_label_font_size)
        ax.set_ylabel("True Positive Rate", fontsize=self.parent.axis_label_font_size)
        ax.set_title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        ax.legend(loc="best", prop={'size': self.parent.plot_font_size})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pr_plot_bin(self, y_score_dec_func, y_test, pos_label: int, plot_widget, name=None) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        if len(y_score_dec_func.shape) > 1 and y_score_dec_func.shape[1] > 1:
            y_score_dec_func = y_score_dec_func[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_score_dec_func, name=name, ax=ax, color='darkorange',
                                                pos_label=pos_label)
        ax.set_title("2-class Precision-Recall curve")
        ax.legend(loc="best", prop={'size': self.parent.plot_font_size})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pr_plot(self, classes: list[int], Y_test, y_score, colors, plot_widget, target_names) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        # The average precision score in multi-label settings
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(classes)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
        display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                         average_precision=average_precision["micro"])
        display.plot(ax=ax, name="Micro-average precision-recall", color="deeppink", linestyle=":", linewidth=3, )
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="deeppink", alpha=0.2)
            ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02), color="deeppink")
        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i], )
            display.plot(ax=ax, name=f"Precision-recall for {target_names[i]}", color=color, linewidth=2)
        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best", prop={'size': self.parent.plot_font_size})
        ax.set_title("Extension of Precision-Recall curve to multi-class")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_dm_plot(self, y_test, x_test, target_names, model, plot_widget) -> None:
        plot_widget.canvas.axes.cla()
        ax = plot_widget.canvas.axes
        # ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, colorbar=False)
        try:
            ax.xaxis.set_ticklabels(target_names, fontsize=self.parent.axis_label_font_size)
            ax.yaxis.set_ticklabels(target_names, fontsize=self.parent.axis_label_font_size)
        except ValueError as err:
            error(err)
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        _ = ax.set_title(f"Confusion Matrix for {model.__class__.__name__}\non the test data")

        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def do_update_shap_plots(self, classificator_type):
        if classificator_type not in self.latest_stat_result \
                or 'target_names' not in self.latest_stat_result[classificator_type]:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.parent.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_means_plot(i, classificator_type)
        self.update_shap_beeswarm_plot(False, i, classificator_type)
        self.update_shap_scatter_plot(False, i, classificator_type)
        self.update_shap_heatmap_plot(False, i, classificator_type)

    def do_update_shap_plots_by_instance(self, classificator_type):
        if classificator_type not in self.latest_stat_result \
                or 'target_names' not in self.latest_stat_result[classificator_type]:
            return
        target_names = self.latest_stat_result[classificator_type]['target_names']
        num = np.where(target_names == self.parent.ui.current_group_shap_comboBox.currentText())
        if len(num[0]) == 0:
            return
        i = int(num[0][0])
        self.update_shap_force(i, classificator_type)
        self.update_shap_force(i, classificator_type, True)
        self.update_shap_decision_plot(i, classificator_type)
        self.update_shap_waterfall_plot(i, classificator_type)

    # endregion

    # region SHAP
    def update_shap_means_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_means
            case 'QDA':
                plot_widget = self.parent.ui.qda_shap_means
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_means
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_means
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_means
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_means
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_means
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_means
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_means
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_means
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_means
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_means
            case _:
                return

        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass

        if classificator_type not in self.latest_stat_result:
            return
        result = self.latest_stat_result[classificator_type]
        if 'shap_values' not in result:
            return
        shap_values = result['shap_values']
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if isinstance(shap_values, list) and not binary:
            shap_values = shap_values[class_i]
        elif not isinstance(shap_values, list) and len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.bar(shap_values, show=False, max_display=20, ax=fig.gca(), fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_beeswarm_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_beeswarm
            case 'QDA':
                plot_widget = self.parent.ui.qda_shap_beeswarm
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_beeswarm
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_beeswarm
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_beeswarm
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_beeswarm
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_beeswarm
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_beeswarm
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_beeswarm
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_beeswarm
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_beeswarm
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_beeswarm
            case _:
                return

        if self.parent.plt_style is None:
            plt.style.use(['dark_background'])
        else:
            plt.style.use(self.parent.plt_style)
        if self.parent.ui.sun_Btn.isChecked():
            color = None
        else:
            color = plt.get_cmap("cool")
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if classificator_type not in self.latest_stat_result:
            return
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.beeswarm(shap_values, show=False, color=color, max_display=20, ax=fig.gca(), fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)
        plt.close('all')

    def update_shap_scatter_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_scatter
            case 'QDA':
                plot_widget = self.parent.ui.qda_shap_scatter
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_scatter
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_scatter
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_scatter
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_scatter
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_scatter
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_scatter
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_scatter
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_scatter
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_scatter
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_scatter
            case _:
                return

        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        current_feature = self.parent.ui.current_feature_comboBox.currentText()
        cmap = None if self.parent.ui.sun_Btn.isChecked() else plt.get_cmap("cool")
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if binary:
            class_i = 0
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        if current_feature not in shap_values.feature_names:
            return
        ct = self.parent.ui.coloring_feature_comboBox.currentText()
        color = shap_values if ct == '' else shap_values[:, ct]
        shap.plots.scatter(shap_values[:, current_feature], color=color, show=False, cmap=cmap, ax=fig.gca(),
                           axis_color=self.parent.plot_text_color.name())
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass

    def update_shap_heatmap_plot(self, binary: bool, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_heatmap
            case 'QDA':
                plot_widget = self.parent.ui.qda_shap_heatmap
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_heatmap
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_heatmap
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_heatmap
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_heatmap
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_heatmap
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_heatmap
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_heatmap
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_heatmap
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_heatmap
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_heatmap
            case _:
                return

        if self.parent.plt_style is None:
            plt.style.use(['dark_background'])
        else:
            plt.style.use(self.parent.plt_style)
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if classificator_type not in self.latest_stat_result:
            return
        if 'shap_values' not in self.latest_stat_result[classificator_type]:
            return
        shap_values = self.latest_stat_result[classificator_type]['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        shap.plots.heatmap(shap_values, show=False, max_display=20, ax=fig.gca(), fig=fig)
        self.parent.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_shap_force(self, class_i: int = 0, classificator_type: str = 'LDA', full: bool = False) -> None:
        if classificator_type == 'QDA':
            return
        if classificator_type not in self.latest_stat_result:
            warning(classificator_type + ' not in self.latest_stat_result. def update_shap_force')
            return
        model = self.get_current_dataset_type_cb()
        if model is None:
            return

        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = 0
        else:
            sample_id = model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if 'explainer' not in result and 'expected_value' not in result:
            return
        elif 'expected_value' in result:
            expected_value = result['expected_value']
        elif 'explainer' in result:
            expected_value = result['explainer'].expected_value
        else:
            return
        if 'shap_values_legacy' not in result:
            return
        shap_values = result['shap_values_legacy']
        if 'X_display' not in result:
            return
        X_display = result['X_display']
        if isinstance(X_display, DataFrame) and X_display.empty or not isinstance(X_display, DataFrame):
            return
        if full and isinstance(shap_values, list):
            shap_v = shap_values[class_i]
        elif isinstance(shap_values, list):
            shap_v = shap_values[class_i][sample_id]
        elif full:
            shap_v = shap_values
        else:
            shap_v = shap_values[sample_id]
        if full:
            x_d = X_display.iloc[:, 2:]
        else:
            x_d = X_display.iloc[:, 2:].loc[sample_id]
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        if (not full and shap_v.shape[0] != len(x_d.values)) or (full and shap_v[0].shape[0] != len(x_d.loc[0].values)):
            err = 'Force plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        try:
            force_plot = shap.force_plot(expected_value, shap_v, x_d, feature_names=result['feature_names'])
        except ValueError:
            error('failed update_shap_force')
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        if full:
            self.latest_stat_result[classificator_type]['shap_html_full'] = shap_html
        else:
            self.latest_stat_result[classificator_type]['shap_html'] = shap_html

    def update_shap_decision_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_decision
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_decision
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_decision
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_decision
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_decision
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_decision
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_decision
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_decision
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_decision
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_decision
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_decision
            case _:
                return
        model = self.get_current_dataset_type_cb()
        if model is None:
            return
        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = None
        else:
            sample_id = model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        if 'explainer' not in result and 'expected_value' not in result:
            return
        elif 'expected_value' in result:
            expected_value = result['expected_value']
        elif 'explainer' in result:
            expected_value = result['explainer'].expected_value
        else:
            return
        if 'shap_values_legacy' not in result:
            return
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        shap_values = result['shap_values_legacy']
        if 'X_display' not in result:
            return
        X_display = result['X_display']
        if isinstance(X_display, DataFrame) and X_display.empty or not isinstance(X_display, DataFrame):
            return
        misclassified = result['y_train_plus_test'] != result['y_pred']
        title = 'all'
        if sample_id is None and isinstance(shap_values, list):
            shap_v = shap_values[class_i]
        elif isinstance(shap_values, list):
            shap_v = shap_values[class_i][sample_id]
        elif sample_id is None:
            shap_v = shap_values
        else:
            shap_v = shap_values[sample_id]

        if sample_id is None:
            x_d = X_display.iloc[:, 2:]
        else:
            x_d = X_display.iloc[:, 2:].loc[sample_id]
            misclassified = misclassified[sample_id]
            title = current_instance
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[class_i]
        if (sample_id is not None and shap_v.shape[0] != len(x_d.values)) \
                or (sample_id is None and shap_v[0].shape[0] != len(x_d.loc[0].values)):
            err = 'Decision plot не смог обновиться. Количество shap_values features != количеству X features.' \
                  ' Возможно была изменена таблица с обучающими данными.' \
                  ' Нужно пересчитать %s' % classificator_type
            print(err)
            error(err)
            return
        feature_display_range_max = -self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else None
        try:
            shap.plots.decision(expected_value, shap_v, x_d, title=title, ignore_warnings=True,
                                feature_display_range=slice(None, feature_display_range_max, -1),
                                highlight=misclassified, ax=ax, fig=fig)
        except ValueError:
            error(classificator_type + ' decision plot trouble. {!s}'.format(ValueError))
            return
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_shap_waterfall_plot(self, class_i: int = 0, classificator_type: str = 'LDA') -> None:
        if classificator_type not in self.latest_stat_result:
            return
        match classificator_type:
            case 'LDA':
                plot_widget = self.parent.ui.lda_shap_waterfall
            case 'QDA':
                plot_widget = self.parent.ui.qda_shap_waterfall
            case 'Logistic regression':
                plot_widget = self.parent.ui.lr_shap_waterfall
            case 'NuSVC':
                plot_widget = self.parent.ui.svc_shap_waterfall
            case 'Nearest Neighbors':
                plot_widget = self.parent.ui.nearest_shap_waterfall
            case 'GPC':
                plot_widget = self.parent.ui.gpc_shap_waterfall
            case 'Decision Tree':
                plot_widget = self.parent.ui.dt_shap_waterfall
            case 'Naive Bayes':
                plot_widget = self.parent.ui.nb_shap_waterfall
            case 'Random Forest':
                plot_widget = self.parent.ui.rf_shap_waterfall
            case 'AdaBoost':
                plot_widget = self.parent.ui.ab_shap_waterfall
            case 'MLP':
                plot_widget = self.parent.ui.mlp_shap_waterfall
            case 'XGBoost':
                plot_widget = self.parent.ui.xgboost_shap_waterfall
            case _:
                return
        model = self.get_current_dataset_type_cb()
        if model.rowCount() == 0:
            return
        q_res = model.dataframe()
        features_names = list(q_res.columns[2:])
        n_features = len(features_names)
        model = self.get_current_dataset_type_cb()
        if model is None:
            return
        current_instance = self.parent.ui.current_instance_combo_box.currentText()
        if current_instance == '':
            sample_id = 0
        else:
            sample_id = model.idx_by_column_value('Filename', current_instance)
        result = self.latest_stat_result[classificator_type]
        if 'model' not in result:
            return
        model = result['model']
        classes = model.classes_
        binary = len(classes) == 2
        if binary:
            class_i = 0
        fig = plot_widget.canvas.figure
        try:
            fig.clear()
        except ValueError:
            pass
        ax = fig.gca()
        try:
            ax.clear()
        except ValueError:
            pass
        if 'shap_values' not in result:
            return
        shap_values = result['shap_values']
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., class_i]
        max_display = self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else n_features
        try:
            shap.plots.waterfall(shap_values[sample_id], max_display, ax=ax, fig=fig)
        except ValueError:
            error(classificator_type + ' waterfall plot trouble. {!s}'.format(ValueError))
            return
        # self.set_canvas_colors(plot_widget.canvas)
        try:
            plot_widget.canvas.figure.tight_layout()
        except ValueError:
            pass
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.resize(plot_widget.width() + 1, plot_widget.height() + 1)
        plot_widget.resize(plot_widget.width() - 1, plot_widget.height() - 1)

    def update_force_single_plots(self, clasificator_type: str = '') -> None:
        cl_types = [('LDA', self.parent.ui.lda_force_single), ('Logistic regression', self.parent.ui.lr_force_single),
                    ('NuSVC', self.parent.ui.svc_force_single),
                    ('Nearest Neighbors', self.parent.ui.nearest_force_single),
                    ('GPC', self.parent.ui.gpc_force_single), ('Decision Tree', self.parent.ui.dt_force_single),
                    ('Naive Bayes', self.parent.ui.nb_force_single), ('Random Forest', self.parent.ui.rf_force_single),
                    ('AdaBoost', self.parent.ui.ab_force_single), ('MLP', self.parent.ui.mlp_force_single),
                    ('XGBoost', self.parent.ui.xgboost_force_single)]
        for cl_type, plot_widget in cl_types:
            if clasificator_type != '' and clasificator_type != cl_type:
                continue
            if cl_type not in self.latest_stat_result:
                continue
            if 'shap_html' in self.latest_stat_result[cl_type]:
                shap_html = self.latest_stat_result[cl_type]['shap_html']
                if self.parent.ui.sun_Btn.isChecked():
                    shap_html = re.sub(r'#ffe', "#000", shap_html)
                    shap_html = re.sub(r'#001', "#fff", shap_html)
                else:
                    shap_html = re.sub(r'#000', "#ffe", shap_html)
                    shap_html = re.sub(r'#fff', "#001", shap_html)
                plot_widget.setHtml(shap_html)
                plot_widget.page().setBackgroundColor(QColor(self.parent.plot_background_color))
                plot_widget.reload()
                self.latest_stat_result[cl_type]['shap_html'] = shap_html
            else:
                error(cl_type + ' shap_html not in self.latest_stat_result')

    def update_force_full_plots(self, clas_type: str = '') -> None:
        cl_types = [('LDA', self.parent.ui.lda_force_full), ('Logistic regression', self.parent.ui.lr_force_full),
                    ('NuSVC', self.parent.ui.svc_force_full), ('Nearest Neighbors', self.parent.ui.nearest_force_full),
                    ('GPC', self.parent.ui.gpc_force_full), ('Decision Tree', self.parent.ui.dt_force_full),
                    ('Naive Bayes', self.parent.ui.nb_force_full), ('Random Forest', self.parent.ui.rf_force_full),
                    ('AdaBoost', self.parent.ui.ab_force_full), ('MLP', self.parent.ui.mlp_force_full),
                    ('XGBoost', self.parent.ui.xgboost_force_full)]
        for cl_type, plot_widget in cl_types:
            if clas_type != '' and clas_type != cl_type:
                continue
            if cl_type not in self.latest_stat_result:
                continue
            if 'shap_html_full' in self.latest_stat_result[cl_type]:
                shap_html = self.latest_stat_result[cl_type]['shap_html_full']
                if self.parent.ui.sun_Btn.isChecked():
                    shap_html = re.sub(r'#ffe', "#000", shap_html)
                    shap_html = re.sub(r'#001', "#fff", shap_html)
                else:
                    shap_html = re.sub(r'#000', "#ffe", shap_html)
                    shap_html = re.sub(r'#fff', "#001", shap_html)
                plot_widget.setHtml(shap_html)
                plot_widget.page().setBackgroundColor(QColor(self.parent.plot_background_color_web))
                plot_widget.reload()
                self.latest_stat_result[cl_type]['shap_html_full'] = shap_html
            else:
                error(cl_type + ' shap_html_full not in self.latest_stat_result')

    # endregion

    def update_plots(self, cl_type) -> None:
        if cl_type not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result[cl_type]
        model = model_results['model']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        target_names = model_results['target_names']
        y_test = model_results['y_test']
        y_score = model_results['y_score']
        y_onehot_test = model_results['y_onehot_test']
        if 'y_score_dec_func' not in model_results:
            y_score_dec_func = y_score
        else:
            y_score_dec_func = model_results['y_score_dec_func']
        if 'use_pca' not in model_results:
            use_pca = True
        else:
            use_pca = model_results['use_pca']
        y_pred_2d = model_results['y_pred_2d']
        y_test_bin = model_results['y_test_bin']
        model_2d = model_results['model_2d']
        y_pred = model_results['y_pred']
        explained_variance_ratio = None
        if 'explained_variance_ratio' in model_results:
            explained_variance_ratio = model_results['explained_variance_ratio']
        classes = model.classes_
        plt_colors = self._get_plot_colors(classes)
        binary = len(classes) == 2
        if cl_type == 'LDA':
            if model_results['features_in_2d'].shape[1] == 1:
                self.update_lda_scores_plot_1d(classes, y_train_plus_test, features_in_2d)
            else:
                self.update_lda_scores_plot_2d(features_in_2d, y_train_plus_test, y_pred, model, model_2d)
        else:
            self.update_scores_plot(features_in_2d, y_train_plus_test, y_pred_2d, model,
                                    model_2d, self.params['score_plots'][cl_type], explained_variance_ratio, use_pca)
        if binary:
            self.update_roc_plot_bin(model, model_results['x_test'], y_test, target_names,
                                     self.params['roc_plots'][cl_type])
            self.update_pr_plot_bin(y_score_dec_func, y_test, classes[0], self.params['pr_plots'][cl_type], cl_type)
        else:
            self.update_roc_plot(y_score, y_onehot_test, target_names, plt_colors, self.params['roc_plots'][cl_type])
            self.update_pr_plot(classes, y_test_bin, y_score_dec_func, plt_colors,
                                self.params['pr_plots'][cl_type], target_names)
        if cl_type in self.params['feature_plots']:
            top = self.update_features_plot(model_results['x_train'], model, model_results['feature_names'],
                                            target_names, plt_colors, self.params['feature_plots'][cl_type])
            self.top_features[cl_type] = top
        if cl_type == 'Decision Tree':
            update_plot_tree(model.best_estimator_, self.parent.ui.dt_tree_plot_widget,
                             model_results['feature_names'], target_names)
            self.update_features_plot_random_forest(model.best_estimator_, self.parent.ui.dt_features_plot_widget)
        elif cl_type == 'Random Forest':
            update_plot_tree(model.best_estimator_.estimators_[0], self.parent.ui.rf_tree_plot_widget,
                             model_results['feature_names'], target_names)
            self.update_features_plot_random_forest(model.best_estimator_, self.parent.ui.rf_features_plot_widget)
        elif cl_type == 'AdaBoost':
            update_plot_tree(model.estimators_[0], self.parent.ui.ab_tree_plot_widget,
                             model_results['feature_names'], target_names)
            self.update_features_plot_random_forest(model, self.parent.ui.ab_features_plot_widget)
        if cl_type == 'XGBoost' and self.parent.ui.dataset_type_cb.currentText() == 'Deconvoluted':
            self.update_features_plot_xgboost(model.best_estimator_)
        elif cl_type == 'XGBoost':
            self.update_features_plot_random_forest(model.best_estimator_, self.parent.ui.xgboost_features_plot_widget)
        if cl_type == 'XGBoost':
            self.update_xgboost_tree_plot(model.best_estimator_)
        self.update_dm_plot(y_test, model_results['x_test'], target_names, model, self.params['dm_plots'][cl_type])
        self.do_update_shap_plots(cl_type)
        self.do_update_shap_plots_by_instance(cl_type)

    # region LDA

    def update_lda_scores_plot_1d(self, classes: np.ndarray, y: list[int], features_in_2d) -> None:
        main_window = self.parent
        main_window.ui.lda_scores_1d_plot_widget.setVisible(True)
        main_window.ui.lda_scores_2d_plot_widget.setVisible(False)

        items_matches = main_window.lda_scores_1d_plot_item.listDataItems()
        for i in items_matches:
            main_window.lda_scores_1d_plot_item.removeItem(i)
        for i in main_window.lda_1d_inf_lines:
            main_window.lda_scores_1d_plot_item.removeItem(i)
        main_window.lda_1d_inf_lines = []
        min_scores = int(np.round(np.min(features_in_2d))) - 1
        max_scores = int(np.round(np.max(features_in_2d))) + 1
        rng = int((max_scores - min_scores) / .1)
        bottom = np.zeros(rng - 1)
        means = []
        for i in classes:
            scores_class_i = []
            for j, score in enumerate(features_in_2d):
                if y[j] == i:
                    scores_class_i.append(score)
            hist_y, hist_x = np.histogram(scores_class_i, bins=np.linspace(min_scores, max_scores, rng))
            centroid = np.mean(scores_class_i)
            means.append(centroid)
            pen_color = main_window.theme_colors['plotText']
            brush = main_window.ui.GroupsTable.model().cell_data_by_idx_col_name(i, 'Style')['color']
            bgi = BarGraphItem(x0=hist_x[:-1], x1=hist_x[1:], y0=bottom, height=hist_y, pen=pen_color, brush=brush)
            bottom += hist_y
            inf_line = InfiniteLine(centroid, pen=QColor(brush))

            main_window.ui.lda_scores_1d_plot_widget.addItem(bgi)
            if not np.isnan(centroid):
                main_window.lda_scores_1d_plot_item.addItem(inf_line)
                main_window.lda_1d_inf_lines.append(inf_line)
        if len(main_window.lda_1d_inf_lines) > 1:
            inf_line_mean = InfiniteLine(np.mean(means), pen=QColor(main_window.theme_colors['inverseTextColor']))
            main_window.lda_scores_1d_plot_item.addItem(inf_line_mean)
            main_window.lda_1d_inf_lines.append(inf_line_mean)

    def update_lda_scores_plot_2d(self, features_in_2d: np.ndarray, y: list[int], y_pred: list[int], model,
                                  model_2d) -> None:
        """
        @param features_in_2d: transformed 2d
        @param y: true labels
        @param y_pred: predicted labels
        @param model: classificator
        @param model_2d: 2d model classificator
        @return:
        """
        main_window = self.parent
        main_window.ui.lda_scores_2d_plot_widget.canvas.axes.cla()
        main_window.ui.lda_scores_1d_plot_widget.setVisible(False)
        main_window.ui.lda_scores_2d_plot_widget.setVisible(True)
        explained_variance_ratio = model.best_estimator_.explained_variance_ratio_
        classes = model.classes_
        plt_colors = []
        for cls in classes:
            clr = main_window.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter(140).name())
        tp = y == y_pred  # True Positive
        cmap = LinearSegmentedColormap('', None).from_list('', plt_colors)
        DecisionBoundaryDisplay.from_estimator(model_2d, features_in_2d, grid_resolution=1000, eps=.5, cmap=cmap,
                                               ax=main_window.ui.lda_scores_2d_plot_widget.canvas.axes)
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            true_positive_of_class = tp[y == cls]
            x_i = features_in_2d[y == cls]
            x_tp = x_i[true_positive_of_class]
            x_fp = x_i[~true_positive_of_class]
            marker = markers[cls]
            color = main_window.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            # inverted_color = invert_color(color)
            main_window.ui.lda_scores_2d_plot_widget.canvas.axes.scatter(x_tp[:, 0], x_tp[:, 1], marker=marker,
                                                                         color=color, edgecolor='black', s=60)
            main_window.ui.lda_scores_2d_plot_widget.canvas.axes.scatter(x_fp[:, 0], x_fp[:, 1], marker="x", s=60,
                                                                         color=color)
        main_window.ui.lda_scores_2d_plot_widget.canvas.axes.set_xlabel(
            'LD-1 (%.2f%%)' % (explained_variance_ratio[0] * 100), fontsize=main_window.axis_label_font_size)
        main_window.ui.lda_scores_2d_plot_widget.canvas.axes.set_ylabel(
            'LD-2 (%.2f%%)' % (explained_variance_ratio[1] * 100), fontsize=main_window.axis_label_font_size)
        try:
            main_window.ui.lda_scores_2d_plot_widget.canvas.draw()
        except ValueError:
            pass
        main_window.ui.lda_scores_2d_plot_widget.canvas.figure.tight_layout()

    # endregion

    # region Random Forest
    def update_features_plot_random_forest(self, model, plot_widget) -> None:
        ax = plot_widget.canvas.axes
        ax.cla()
        if self.parent.ui.dataset_type_cb.currentText() == 'Deconvoluted':
            mdi_importances = Series(model.feature_importances_, index=model.feature_names_in_).sort_values(
                ascending=True)
            mdi_importances = mdi_importances[mdi_importances > 0]
            ax.barh(mdi_importances.index, mdi_importances, color=self.parent.theme_colors['primaryColor'])
            ax.set_xlabel('MDI importance', fontsize=self.parent.axis_label_font_size)
            ax.set_ylabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=self.parent.axis_label_font_size)
        else:
            k = []
            for i in model.feature_names_in_:
                k.append(float(i.replace('k', '').replace('_a', '').replace('_x0', '')))
            ser = Series(model.feature_importances_, index=k)
            ax.plot(ser)
            ax.set_xlabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=self.parent.axis_label_font_size)
            ax.set_ylabel('MDI importance', fontsize=self.parent.axis_label_font_size)

        if plot_widget == self.parent.ui.rf_features_plot_widget:
            ax.set_title("Random Forest Feature Importances (MDI)")
        elif plot_widget == self.parent.ui.ab_features_plot_widget:
            ax.set_title("AdaBoost Feature Importances (MDI)")
        else:
            ax.set_title("XGBoost Feature Importances (MDI)")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region PCA

    def update_pca_plots(self) -> None:
        """
        Update all 'PCA plots and fields
        @return: None
        """
        if 'PCA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['PCA']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        explained_variance_ratio = model_results['explained_variance_ratio']

        self.update_scores_plot_pca(features_in_2d, y_train_plus_test, self.parent.ui.pca_scores_plot_widget,
                                    explained_variance_ratio, model_results['target_names'])
        if 'loadings' in model_results:
            self.parent.ui.pca_features_table_view.model().set_dataframe(model_results['loadings'])
            self.update_pca_loadings_plot(model_results['loadings'], explained_variance_ratio)

    def update_scores_plot_pca(self, features_in_2d: np.ndarray, y: list[int], plot_widget,
                               explained_variance_ratio, target_names) -> None:
        """
        @param target_names:
        @param plot_widget:
        @param features_in_2d: transformed 2d
        @param y: fact labels
        @param explained_variance_ratio:
        @return:
        """
        ax = plot_widget.canvas.axes
        ax.cla()
        classes = np.unique(y)
        plt_colors = []
        for cls in classes:
            clr = self.parent.get_color_by_group_number(cls)
            plt_colors.append(clr.lighter().name())
        markers = ['o', 's', 'D', 'h', 'H', 'p', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*']
        for i, cls in enumerate(classes):
            x_i = features_in_2d[y == cls]
            mrkr = markers[cls]
            color = self.parent.ui.GroupsTable.model().cell_data_by_idx_col_name(cls, 'Style')['color'].name()
            ax.scatter(x_i[:, 0], x_i[:, 1], marker=mrkr, color=color,
                       edgecolor='black', s=60, label=target_names[i])
        if plot_widget == self.parent.ui.pca_scores_plot_widget:
            ax.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                          fontsize=self.parent.plot_font_size)
            ax.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                          fontsize=self.parent.plot_font_size)
        else:
            ax.set_xlabel('PLS-DA-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                          fontsize=self.parent.plot_font_size)
            ax.set_ylabel('PLS-DA-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                          fontsize=self.parent.plot_font_size)
        ax.legend(loc="best", prop={'size': self.parent.plot_font_size - 2})
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_pca_loadings_plot(self, loadings, explained_variance_ratio) -> None:
        plot_widget = self.parent.ui.pca_loadings_plot_widget
        plot_widget.canvas.axes.cla()
        features_names = list(loadings.axes[0])
        plot_widget.canvas.axes.scatter(loadings['PC1'], loadings['PC2'],
                                        color=self.parent.theme_colors['primaryColor'], s=60)
        for i, txt in enumerate(features_names):
            plot_widget.canvas.axes.annotate(txt, (loadings['PC1'][i], loadings['PC2'][i]))
        plot_widget.canvas.axes.set_xlabel('PC-1 (%.2f%%)' % (explained_variance_ratio[0] * 100),
                                           fontsize=self.parent.plot_font_size)
        plot_widget.canvas.axes.set_ylabel('PC-2 (%.2f%%)' % (explained_variance_ratio[1] * 100),
                                           fontsize=self.parent.plot_font_size)
        plot_widget.canvas.axes.set_title("PCA Loadings")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region PLS DA

    def update_plsda_plots(self) -> None:
        """
        Update all PLS-DA plots and fields
        @return: None
        """
        if 'PLS-DA' not in self.latest_stat_result:
            return
        model_results = self.latest_stat_result['PLS-DA']
        y_train_plus_test = model_results['y_train_plus_test']
        features_in_2d = model_results['features_in_2d']
        explained_variance_ratio = model_results['explained_variance_ratio']

        self.update_scores_plot_pca(features_in_2d, y_train_plus_test, self.parent.ui.plsda_scores_plot_widget,
                                    explained_variance_ratio, model_results['target_names'])
        if 'vips' in model_results:
            vips = model_results['vips']
            df = DataFrame(list(zip(model_results['feature_names'], vips)), columns=['feature', 'VIP'])
            self.parent.ui.plsda_vip_table_view.model().set_dataframe(df)
            self.update_vip_plot(model_results['vips'], model_results['feature_names'])

    def update_vip_plot(self, vips, features_names) -> None:
        plot_widget = self.parent.ui.plsda_vip_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        if self.parent.ui.dataset_type_cb.currentText() == 'Deconvoluted':
            ser = Series(vips, index=features_names).sort_values(ascending=True)
            ax.barh(ser.index, ser, color=self.parent.theme_colors['primaryColor'])
            ax.set_xlabel('VIP', fontsize=self.parent.axis_label_font_size)
            ax.set_ylabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=self.parent.axis_label_font_size)
        else:
            k = []
            for i in features_names:
                k.append(float(i.replace('k', '')))
            ser = Series(vips, index=k)
            ax.plot(ser)
            ax.set_xlabel('Raman shift, cm\N{superscript minus}\N{superscript one}',
                          fontsize=self.parent.axis_label_font_size)
            ax.set_ylabel('VIP', fontsize=self.parent.axis_label_font_size)
        ax.set_title("Variable Importance in the Projection (VIP)")
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    # endregion

    # region XGBoost
    def update_features_plot_xgboost(self, model) -> None:
        plot_widget = self.parent.ui.xgboost_features_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        max_num_features = self.parent.ui.feature_display_max_spinBox.value() \
            if self.parent.ui.feature_display_max_checkBox.isChecked() else None
        xgboost.plot_importance(model, ax, max_num_features=max_num_features, grid=False)
        try:
            plot_widget.canvas.draw()
        except ValueError:
            pass
        plot_widget.canvas.figure.tight_layout()

    def update_xgboost_tree_plot(self, model, idx: int = 0) -> None:
        plot_widget = self.parent.ui.xgboost_tree_plot_widget
        ax = plot_widget.canvas.axes
        ax.cla()
        try:
            xgboost.plot_tree(model, ax=ax, num_trees=idx, yes_color=self.parent.theme_colors['secondaryDarkColor'],
                              no_color=self.parent.theme_colors['primaryColor'])
            plot_widget.canvas.draw()
        except:
            error('update_xgboost_tree_plot')
            return
        plot_widget.canvas.figure.tight_layout()
    # endregion
