from rewriter.utils.bleu import BLEUScore

metric = BLEUScore()
result = "This is a cat."
target = ["He be great.", "That is a cat."]
metric.append(result, target)
print(metric.score())