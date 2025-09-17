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
    A = np.array(inputs.get("matrix", []), dtype=float)
    row_sums = A.sum(axis=1) if A.ndim==2 and A.size else []
    cap = ("행 정규화 혼동행렬: (행=실제, 값=해당 행 내 비율·재현율 관점)"
        if len(row_sums) and np.all((row_sums>0.95)&(row_sums<1.05))
        else "절대 카운트 혼동행렬: 각 셀은 예측×실제 표본 수(분포 반영)")
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.figtext(0.5, 0.005, cap, ha="center", va="bottom", fontsize=9)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("confusion_matrix", ["matrix","labels"], ["title"], render_confusion_matrix))
