echo "start running deepspeech test 1......"
PATH=/usr/local/cuda-8.0/bin/:$PATH LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH deepspeech models/output_graph.pb  ./audio/8842-304647-0002.wav  models/alphabet.txt models/lm.binary  models/trie


echo "start running deepspeech test 2......"
PATH=/usr/local/cuda-8.0/bin/:$PATH LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH deepspeech models/output_graph.pb  ./audio/8842-304647-0010.wav  models/alphabet.txt models/lm.binary  models/trie


