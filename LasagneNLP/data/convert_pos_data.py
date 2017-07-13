import codecs


f1 = codecs.open('POS.txt', 'r', 'utf-8')
f2 = codecs.open('POS_NEW.txt', 'w', 'utf-8')
cnt = 0
for line in f1:
    if line != '\n':
        line = line.strip().split('\t')
        line = line[:1] + [u'_', u'_'] + line[1:] + [u'_']
        f2.write('\t'.join(line) + '\n')
    else:
        f2.write('\n')
        cnt += 1
print cnt
f1.close()
f2.close()