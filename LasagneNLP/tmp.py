import codecs


def load_word2vec(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    word2vec_list = []
    f.readline()
    cnt = 0
    for line in f:
        cnt += 1
        try:
            word2vec_list.append(line.split()[0])
        except:
            print cnt
            print line
    f.close()
    return word2vec_list


if __name__ == '__main__':
    load_word2vec('data/Vie_Skip_Gram_300.txt')