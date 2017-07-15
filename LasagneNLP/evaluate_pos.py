import codecs
from sklearn.metrics import f1_score

f = codecs.open('output/bi-lstm-cnn-crf-pos/dev15', 'r', 'utf-8')
predict_list = []
test_list = []
for line in f:
    if line != '\n':
        line = line.split()
        test_list.append(line[1])
        predict_list.append(line[2])
print f1_score(test_list, predict_list, average='micro')