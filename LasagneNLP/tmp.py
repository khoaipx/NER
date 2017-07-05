import codecs


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


if __name__ == '__main__':
    #load_word2vec('data/Vie_Skip_Gram_300.txt')
    create_word2vec_new('data/Vie_Skip_Gram_300.txt', 'data/Vie_Skip_Gram_300_new.txt')