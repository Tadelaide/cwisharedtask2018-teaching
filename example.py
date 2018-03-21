from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
import timeit

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.devset)
    #gold_label is the binary result 
    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    start = timeit.default_timer()
    execute_demo('english')
    englishTime = timeit.default_timer() - start
    print('To complete English part, it take ',englishTime,' seconds')
    start1 = timeit.default_timer()
    execute_demo('spanish')
    spanishTime = timeit.default_timer() - start1
    print('To complete English part, it take ',spanishTime,' seconds')


