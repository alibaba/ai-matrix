mkdir -p log
make sgemm
./bin/sgemm 64 10240 >log/sgemm.log
