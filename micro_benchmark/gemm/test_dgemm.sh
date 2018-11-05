mkdir -p log
make dgemm
./bin/dgemm 64 10240 >log/dgemm.log
