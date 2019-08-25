sentence_file_path = '/home/tim/Documents/NLP/electronics/electronics_large.csv'
sentence_remapped_file_path = '/home/tim/Documents/NLP/electronics/electronics_balanced_large.csv'

label_cnt = {}


def fiveToThreeClasses(label):
    if label == '1' or label == '2':
        return -1
    elif label == '3':
        return 0
    else:
        return 1


def balance_electronics_dataset():
    """
    Balance the electronics data set by determining the class (pos, neut, neg) with the smallest number of reviews
    and then only keeping this number of reviews from the three categories.
    :return:
    """
    with open(sentence_file_path) as in_file, open(sentence_remapped_file_path, 'a') as out_file:
        next(in_file)
        linesPos = []
        linesNeg = []
        linesNeu = []
        for line in in_file:
            asdf = line.split(',', maxsplit=2)
            label = fiveToThreeClasses(asdf[1])
            if label == -1:
                linesNeg.append(line)
            elif label == 0:
                linesNeu.append(line)
            else:
                linesPos.append(line)

        smaller_class = min(len(linesPos), len(linesNeu), len(linesNeg))
        print("Set the number of samples per category to: %s" % smaller_class)
        for i in range(smaller_class):
            out_file.write(linesPos[i])
            out_file.write(linesNeg[i])
            out_file.write(linesNeu[i])

balance_electronics_dataset()