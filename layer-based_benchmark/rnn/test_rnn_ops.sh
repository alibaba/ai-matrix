mkdir -p log
python rnn_ops.py -op_name lstm -times 50 -interval 1 -min_bs 16 -max_bs 32 -stride_bs 16 -min_sl 32 -max_sl 64 -stride_sl 32 -min_units 256 -max_units 2560 -stride_units 256
sleep 1

python rnn_ops.py -op_name gru -times 50 -interval 1 -min_bs 16 -max_bs 32 -stride_bs 16 -min_sl 32 -max_sl 64 -stride_sl 32 -min_units 256 -max_units 2560 -stride_units 256

sleep 1
python rnn_ops.py -op_name rnn -times 50 -interval 1 -min_bs 16 -max_bs 32 -stride_bs 16 -min_sl 32 -max_sl 64 -stride_sl 32 -min_units 256 -max_units 2560 -stride_units 256
