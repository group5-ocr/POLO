# 혼동행렬(Confusion Matrix) 히트맵. 각 셀에 값 주석 표시.
import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_confusion_matrix(inputs, out_path):
    mat = inputs.get("matrix", [])
    labels = inputs.get("labels", [])
    A = np.array(mat, dtype=float)
    plt.figure(figsize=(5.2,4.8))
    plt.imshow(A, interpolation="nearest")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(inputs.get("title","Confusion Matrix"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("confusion_matrix", ["matrix","labels"], ["title"], render_confusion_matrix))
