import codecs


def read_data(raw_data):
    sentences = list()
    temp = list()
    for line in codecs.open(raw_data, 'r', 'utf8'):
        line = line.strip()
        if line == '':
            sentences.append(temp)
            temp = list()
        else:
            temp.append(line)
    return sentences


def write_file(sentences, out_file):
    writer = codecs.open(out_file, 'w', 'utf8')
    for sent in sentences:
        for i, line in enumerate(sent):
            tokens = line.split('\t')
            writer.write(str(i+1) + ' ')
            writer.write(' '.join(tokens[:-1]))
            writer.write('\n')
        writer.write('\n')
    writer.close()


def main():
    raw_data = '/home/khoaipx/Downloads/vlsp_corpus.txt'
    sentences = read_data(raw_data)
    print 'Train: ', len(sentences[:14861])
    write_file(sentences[:14861], 'train.txt')
    print 'Dev: ', len(sentences[14861:16861])
    write_file(sentences[14861:16861], 'dev.txt')
    print 'Test: ', len(sentences[16861:])
    write_file(sentences[16861:], 'test.txt')


if __name__ == '__main__':
    main()
