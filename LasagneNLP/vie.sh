THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python bi_lstm_cnn_crf.py --fine_tune --embedding word2vec --oov embedding --update momentum --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout --train "data/train.txt" --dev "data/dev.txt" --test "data/test.txt" --embedding_dict "data/Vie_Skip_Gram_300_new.txt" --patience 5
