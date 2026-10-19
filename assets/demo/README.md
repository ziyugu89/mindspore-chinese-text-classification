# Demo 结果图（提交前请补齐）

训练完成后，脚本 / notebook 会自动把下列图片写到本目录：

| 文件 | 含义 |
| --- | --- |
| `accuracy_loss_curves.png` | 训练/测试 Loss 与 Accuracy 曲线（含 80% 参考线） |
| `confusion_matrix.png` | 测试集混淆矩阵 |
| `sample_predictions.png` | 若干条样例：真值 / 预测 |

## 如何生成

在项目根目录、conda 环境 `ms_docs_try` 中：

```bat
python train_eval.py --epochs 5 --model textcnn
```

或打开 `notebooks/train_eval.ipynb`，**Restart & Run All**。

## 提交提示

- 把生成的 PNG 一并推到 GitHub（若被 `.gitignore` 忽略，可临时去掉对应规则或强制 `git add -f`）。
- README 的「实验结果」一节用 Markdown 引用这些图片。
- **不要**制作教学视频。
