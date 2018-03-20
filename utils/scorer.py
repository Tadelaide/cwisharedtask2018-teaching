import sklearn


def report_score(gold_labels, predicted_labels, detailed=True):
    macro_F1 = sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro')
    print("macro-F1: {:.2f}".format(macro_F1))
    if detailed:
        scores = sklearn.metrics.precision_recall_fscore_support(gold_labels, predicted_labels)
        #{:^10}.format() 一个标准表达的例子
        print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
        print('-' * 50)
        #{:^10.2f}.format() 一个标准表达两位的 float 的例子
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
    print()
