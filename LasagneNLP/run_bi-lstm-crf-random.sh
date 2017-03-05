THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python bi_lstm_crf.py --fine_tune --embedding random --oov random --update momentum --batch_size 10 --num_units 200 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout --train "data/train.txt" --dev "data/dev.txt" --test "data/test.txt" --embedding_dict "data/Vie_Skip_Gram_300.txt.gz" --patience 5
