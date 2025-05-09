import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import difflib

def optimal_threshold(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = y_pred.ravel()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Calculate Youden's J statistic
    J = tpr - fpr

    # Find the optimal threshold
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def heatmap(data, output_dir='./heatmaps'):
    """
    Create the heatmap of attention weight
    """
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(data, cmap='viridis', linewidths=0.5)

    plt.title('Heatmap of Distinct Row and Column Features', fontsize=24)
    plt.xlabel('Column Features', fontsize=24)
    plt.ylabel('Row Features', fontsize=24)

    # Adjust the appearance for clarity
    plt.xticks(rotation=90, ha='right', fontsize=24)  # Rotate the x-axis labels for better readability
    plt.yticks(rotation=0, fontsize=24)  # Keep the y-axis labels horizontal

    # Display the heatmap
    plt.show()
    plt.savefig(output_dir)

def find_first_element_in_list(element, array_list, begin_idx=0):
        for idx in range(begin_idx, len(array_list)):
            if array_list[idx] == element:
                return idx
        return -1

def find_diff_indexes_in_arrays(array, array_pair, item_real_begin_index=0, item_pair_real_begin_index=0):
    # convert int array to strin array
    array = [str(a) for a in array]
    array_pair = [str(a) for a in array_pair]

    # Creating a Differ object
    d = difflib.Differ()

    # Calculating the difference
    diff = list(d.compare(array, array_pair))
    # print(diff)
    # # Extracting the index of changing words
    # changes = [i for i, word in enumerate(diff) if word.startswith("+ ") or word.startswith("- ")]

    i, j = 0, 0
    old_diff = []
    new_diff = []
    for idx, word in enumerate(diff):
        if word.startswith("+"):
            new_diff.append(i)
            i += 1
        elif word.startswith("-"):
            old_diff.append(j)
            j += 1
        else:
            i += 1
            j += 1

    return old_diff, new_diff

def process_attention_weight(tokenizer, input_ids):
    attetion_masks = []
    for i, ids in enumerate(input_ids):
        sep_idx1 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=0)
        array = ids[1:sep_idx1]

        sep_idx2 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=sep_idx1+1)
        array_pair = ids[sep_idx1+1:sep_idx2]
            
        temp_old_diff, temp_new_diff = find_diff_indexes_in_arrays(array, array_pair)
        old_diff = [idx+1 for idx in temp_old_diff]
        new_diff = [idx+sep_idx1+1 for idx in temp_new_diff]
        changes = old_diff + new_diff

        atte_mask = []
        for idx in range(len(ids)):
            if idx in changes:
                atte_mask.append(1)
            elif ids[idx] == tokenizer.pad_token_id:
                atte_mask.append(0)
            else:
                atte_mask.append(0.5)
        attetion_masks.append(atte_mask)

    return attetion_masks