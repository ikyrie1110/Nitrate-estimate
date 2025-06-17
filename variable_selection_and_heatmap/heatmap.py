import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.cluster import hierarchy
from scipy.stats import pearsonr

def plot_correlation_heatmap(data):
    """
    绘制相关性热力图并显示显著性标注
    """
    corr_matrix = data.corr()
    distance_matrix = 1 - abs(corr_matrix)
    linkage_matrix = hierarchy.linkage(distance_matrix.values[np.triu_indices(len(distance_matrix), 1)], method='ward')
    order = hierarchy.leaves_list(linkage_matrix)
    corr_matrix_ordered = corr_matrix.iloc[order, order]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix_ordered, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True, cbar_kws={'label': '相关性'})
    ax.set_facecolor('white')
    ax = plt.gca()

    for i in range(len(corr_matrix_ordered.columns)):
        for j in range(len(corr_matrix_ordered.columns)):
            if i == j:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=True, facecolor='white', edgecolor='black', linewidth=0.5))
                col_name = corr_matrix_ordered.columns[i]
                ax.text(j + 0.5, i + 0.5, col_name, ha='center', va='center', color='red', fontsize=12)
            elif i < j:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=True, facecolor='white', edgecolor='black', linewidth=0.5))
                corr, p_value = pearsonr(data[corr_matrix_ordered.columns[i]], data[corr_matrix_ordered.columns[j]])

                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = ''

                if significance:
                    norm = plt.Normalize(vmin=-1, vmax=1)
                    facecolor = plt.cm.coolwarm(norm(corr))
                    circle = patches.Circle((j + 0.5, i + 0.5), 0.4, facecolor=facecolor, edgecolor='black', linewidth=0.5)
                    ax.add_patch(circle)
                    ax.text(j + 0.5, i + 0.5, significance, ha='center', va='center', color='black', fontsize=10)

    plt.title('相关性热力图（hclust排序，ward.D2聚类）', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    ax.add_patch(patches.Rectangle((0, 0), len(corr_matrix_ordered.columns), len(corr_matrix_ordered.columns), fill=False, edgecolor='black', linewidth=1))
    plt.tight_layout()
    plt.show()
