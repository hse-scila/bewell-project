from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt


def get_metrics(y_test, predicted_y):
    mse = round(mean_squared_error(y_test, predicted_y), 4)
    mae = round(mean_absolute_error(y_test, predicted_y),4)
    r = round(r2_score(y_test, predicted_y), 4)
    pearsonr_corr, pearsonr_corr_p = stats.pearsonr(y_test, predicted_y)
    spearman_corr, spearman_p = stats.spearmanr(y_test, predicted_y)
    print("Report:")
    print(f"mean_squared_error: {mse}\nmean_absolute_error: {mae}\nr2_score {r}")
    print(f"pearsonr_corr: {round(pearsonr_corr, 4)}, pearsonr_p: {round(pearsonr_corr_p, 4)}")
    print(f"spearman_corr: {round(spearman_corr, 4)}, spearman_p: {round(spearman_p, 4)}")
    print()


def plot_two_distributions(targets, predicted):
    plt.hist(targets, bins=len(set(targets)), color="blue", label='target', )
    plt.hist(predicted, bins=len(set(targets)), color="red", label='predict')
    plt.legend()
    plt.ylabel("Frequence of value in dataset")
    plt.xlabel('Differrent score values')
    plt.show()
    plt.scatter(predicted, targets)
    plt.ylabel("Target")
    plt.xlabel('Predict')
    plt.show()
