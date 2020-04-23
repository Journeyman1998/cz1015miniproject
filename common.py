import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Plot confusion matrix from confusion_matrix object
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('Predicted', size = 18)
    plt.xlabel('Actual', size = 18)


# TP FP
# FN TN
def matrix_data(cm, print_values = True):
    tp_dg1 = cm[0][0]
    tp_dg2 = cm[1][1]
    tp_dg3 = cm[2][2]

    fn_dg1 = cm[1][0] + cm[2][0]
    fn_dg2 = cm[0][1] + cm[2][1]
    fn_dg3 = cm[0][2] + cm[1][2]

    tpr_dg1 = tp_dg1 / (tp_dg1 + fn_dg1)
    tpr_dg2 = tp_dg2 / (tp_dg2 + fn_dg2)
    tpr_dg3 = tp_dg3 / (tp_dg3 + fn_dg3)

    fnr_dg1 = fn_dg1 / (tp_dg1 + fn_dg1)
    fnr_dg2 = fn_dg2 / (tp_dg2 + fn_dg2)
    fnr_dg3 = fn_dg3 / (tp_dg3 + fn_dg3)

    fp_dg1 = cm[0][1] + cm[0][2]
    fp_dg2 = cm[1][0] + cm[1][2]
    fp_dg3 = cm[2][0] + cm[2][1]

    tn_dg1 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
    tn_dg2 = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
    tn_dg3 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]

    fpr_dg1 = fp_dg1 / (fp_dg1 + tn_dg1)
    fpr_dg2 = fp_dg2 / (fp_dg2 + tn_dg2)
    fpr_dg3 = fp_dg3 / (fp_dg3 + tn_dg3)

    tnr_dg1 = tn_dg1 / (fp_dg1 + tn_dg1)
    tnr_dg2 = tn_dg2 / (fp_dg2 + tn_dg2)
    tnr_dg3 = tn_dg3 / (fp_dg3 + tn_dg3)

    precision_dg1 = tp_dg1 / (tp_dg1 + fp_dg1)
    precision_dg2 = tp_dg2 / (tp_dg2 + fp_dg2)
    precision_dg3 = tp_dg3 / (tp_dg3 + fp_dg3)

    f1_dg1 = 2 * ((precision_dg1 * tpr_dg1)/(precision_dg1 + tpr_dg1))
    f1_dg2 = 2 * ((precision_dg2 * tpr_dg2)/(precision_dg2 + tpr_dg2))
    f1_dg3 = 2 * ((precision_dg3 * tpr_dg3)/(precision_dg3 + tpr_dg3))

    p_micro = (tp_dg1 + tp_dg2 + tp_dg1)/((tp_dg1 + fp_dg1) + (tp_dg2 + fp_dg2) + (tp_dg3 + fp_dg3))
    r_micro = (tp_dg1 + tp_dg2 + tp_dg1)/((tp_dg1 + fn_dg1) + (tp_dg2 + fn_dg2) + (tp_dg3 + fn_dg3))
    
    f1_micro = (2 * p_micro * r_micro)/(p_micro + r_micro)
    if(print_values):
        print("True Positive for damage grade 1 : ", tp_dg1)
        print("False Negative for damage grade 1 : ", fn_dg1)
        print("False Positive for damage grade 1 : ", fp_dg1)
        print("True Negative for damage grade 1 : ", tn_dg1)
        print()
        print("True Positive rate for damage grade 1 : ", tpr_dg1)
        print("False Negative rate for damage grade 1 : ", fnr_dg1)
        print("False Positive rate for damage grade 1 : ", fpr_dg1)
        print("True Negative rate for damage grade 1 : ", tnr_dg1)
        print()
        print("Precision for damange grade 1 : ", precision_dg1)
        print("Recall for damange grade 1 : ", tpr_dg1)
        print("F1 score for damage grade 1 : ", f1_dg1)
        print("\n")
        print("True Positive for damage grade 2 : ", tp_dg2)
        print("False Negative for damage grade 2 : ", fn_dg2)
        print("False Positive for damage grade 2 : ", fp_dg2)
        print("True Negative for damage grade 2 : ", tn_dg2)
        print()
        print("True Positive rate for damage grade 2 : ", tpr_dg2)
        print("False Negative rate for damage grade 2 : ", fnr_dg2)
        print("False Positive rate for damage grade 2 : ", fpr_dg2)
        print("True Negative rate for damage grade 2 : ", tnr_dg2)
        print()
        print("Precision for damange grade 2 : ", precision_dg2)
        print("Recall for damange grade 2 : ", tpr_dg2)
        print("F1 score for damage grade 2 : ", f1_dg2)
        print("\n")
        print("True Positive for damage grade 3 : ", tp_dg3)
        print("False Negative for damage grade 3 : ", fn_dg3)
        print("False Positive for damage grade 3 : ", fp_dg3)
        print("True Negative for damage grade 3 : ", tn_dg3)
        print()
        print("True Positive rate for damage grade 3 : ", tpr_dg3)
        print("False Negative rate for damage grade 3 : ", fnr_dg3)
        print("False Positive rate for damage grade 3 : ", fpr_dg3)
        print("True Negative rate for damage grade 3 : ", tnr_dg3)
        print()
        print("Precision for damange grade 3 : ", precision_dg3)
        print("Recall for damange grade 3 : ", tpr_dg3)
        print("F1 score for damage grade 3 : ", f1_dg3)
        print()
        print("F1 micro score is : ", f1_micro)
    return f1_micro


def count_2vars(cols, data):
    """Counting the data against two variables, and calculate the frequency in each bunch of data grouped by first label:
    
    Parameters:
    cols: binary sequence. Each of the element is the label to be counted.
    data: the data to be count
    
    Output:
    a pandas.DataFrame with 4 columns: cols[0], cols[1], "count", "frequency"
    for each row, suppose cols[0] field == A and cols[1] field == B,
    "count" fields contain the count of records which cols[0] field == A and cols[1] field == B
    "frequency" fields contain the frequency of record which cols[1] field == B in the data which cols[0] field == A,
    that is, group the data according the cols[0] field and
    calculate the frequency of each cols[1] field in each group
    """
    grouped = data.groupby([cols[0]])[cols[1]]
    result = (grouped.value_counts()
                     .rename("count")
                     .reset_index()
                     .sort_values(cols[0]))
    freq = (grouped.value_counts(normalize=True)
                     .rename("frequency")
                     .reset_index()
                     .sort_values(cols[0]))
    result["frequency"] = freq["frequency"]
    return result

# Requires categorical data to be encoded into numbers
def plot_bar(data, col, axes):
    """Plot three bar charts about col field and damage_grade on axes_h.
    First chart is the count of records with each combination of value of col and damage_grade
    Second chart is the frequency of damage_grade in each group of data grouped by col field.
    Third chart is the average damage_grade of each group of data grouped by col field.
    
    Parameter:
    col: the field to be plot
    axes: the matplot axes, with at least 3 elements
    
    Output:
    Three graphs described above, plot on axes
    """
    damage_label = 'damage_grade'
    count_data = count_2vars([col, damage_label], data)
    sb.barplot(x=col, y="count", hue=damage_label, data=count_data, ax = axes[0])
    sb.barplot(x=col, y="frequency", hue=damage_label, data=count_data, ax = axes[1])
    for index, row in count_data.iterrows():
        axes[0].text(row[col]+0.27*(row[damage_label]-2), row["count"],
                        row["count"], color='black', ha="center")
        axes[1].text(row[col]+0.27*(row[damage_label]-2), row["frequency"],
                        round(row["frequency"],3), color='black', ha="center")
    sb.barplot(x=col, y=damage_label, data=data21_30, ax = axes[2])
    

# Automatically generates test csv for submission
def genTestLabels(name, building_id, test_pred):
    arr = np.concatenate((building_id[:,None], test_pred[:,None]), axis=1)
    df = pd.DataFrame(arr, columns=['building_id', 'damage_grade'])
    df.to_csv('data/' + name + '.csv', index=False)