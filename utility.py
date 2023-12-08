from collections import defaultdict
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt
import numpy as np

privileged_groups = [{'RACE': 1}]
unprivileged_groups = [{'RACE': 0}]

def test(dataset, model, thresh_arr):
    """
    Function to perform a suite of bias metrics on a trained model.
    test suite includes:
        1. balanced accuracy
        2. average odds difference
        3. disparate impact
        4. statistical parity difference
        5. equal opportunity difference
        6. Theil index

    Args:
        dataset(StandardDataSet): dataset class of representative data to test.
        model(sklearn model): model to be tested against.
        thresh_arr(list): array of thresholds to use in testing.

    Returns:
        defaultdict of metric names and their values when tested against the provided model.
    """
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs

def describe_metrics(metrics, thresh_arr):
    """
    Function to display results of the previous function, test().

    Args:
        metrics(defaultdict): collection of relevant values according to test().
        thresh_arr(list): array of thresholds used in testing.

    Returns:
        None, but prints out the benchmark values for the provided model, according to the
        results of test().
        
    """
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))

def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    """
    Function to plot bias metrics on a dual-axis graph.

    Args:
        x(iterable): The x-axis data values.
        x_name(str): Label for the x-axis.
        y_left(iterable): The left y-axis data values.
        y_left_name(str): Label for the left y-axis.
        y_right(iterable): The right y-axis data values.
        y_right_name(str): Label for the right y-axis.

    Returns:
        None, but plots a visualization for the results of bias metrics.
    """
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)