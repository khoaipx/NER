import codecs
import numpy as np
import cPickle as pickle


def load_word2vec(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    word2vec_list = []
    f.readline()
    cnt = 0
    for line in f:
        cnt += 1
        print cnt
        try:
            word2vec_list.append(line.split()[0])
        except:
            print cnt
            #print line
    f.close()
    print cnt
    return word2vec_list


def create_word2vec_new(filename1, filename2):
    f1 = codecs.open(filename1, 'r', 'utf-8', 'ignore')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    cnt = 0
    for line in f1:
        cnt += 1
        if len(line) > 10:
            f2.write(line)
        else:
            print cnt
    f1.close()
    f2.close()


def dump_word2vec(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    words = []
    vectors = []
    f.readline()
    for line in f:
        try:
            line = line.split()
            words.append(line[0])
            #print line[1:]
            vectors.append([float(i) for i in line[1:]])
        except:
            print line
    vectors = np.asarray(vectors)
    np.save('tmp/vectors', vectors)
    with open('tmp/words.pl', 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dump_word2vec('data/Vie_Skip_Gram_300_new.txt')
