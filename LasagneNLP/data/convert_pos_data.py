import codecs
import collections

standard_set = ['N', 'Np', 'Nc', 'Nu', 'V', 'A', 'P', 'L', 'M', 'R', 'E', 'C', 'I', 'T', 'B', 'Y', 'S', 'X']
f1 = codecs.open('POS.txt', 'r', 'utf-8')
f2 = codecs.open('POS_NEW.txt', 'w', 'utf-8')
cnt = 0
label_set = []
check = 0
cnt1 = 0
for line in f1:
    if line != '\n':
        cnt1 += 1
        line = line.strip().split('\t')
        line = line[:1] + [u'_', u'_'] + line[1:] + [u'_']
        f2.write('\t'.join(line) + '\n')
        label_set.append(line[3])
        if line[3] == u'+':
            print cnt1
    else:
        cnt1 += 1
        f2.write('\n')
        cnt += 1
#label_set = list(set(label_set))
#print len(label_set)
#for item in label_set:
#    if item not in standard_set:
        #print item
print list(set(label_set))
counter = collections.Counter(label_set)
print(counter)

f1.close()
f2.close()