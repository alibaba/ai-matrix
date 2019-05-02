#!/bin/bash

apt-get update

ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.1.0 /usr/lib/x86_64-linux-gnu/libhdf5.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

apt-get install -y libboost-all-dev  gfortran libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev liblmdb-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libhdf5-serial-dev libleveldb-dev libopencv-dev libsnappy-dev

apt-get install -y python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-h5py python-protobuf python-leveldb python-networkx python-nose python-pandas python-gflags cython ipython python-yaml 

dpkg --configure -a
apt-get -f install
apt-get install libhdf5-dev

if [ -f /usr/bin/python ]; then
	rm /usr/bin/python
	sudo rm /usr/bin/python
fi

if [ -f /usr/bin/python3.5 ]; then
	ln -s /usr/bin/python3.5 /usr/bin/python
	sudo ln -s /usr/bin/python3.5 /usr/bin/python
elif [ -f /usr/bin/python3.6 ]; then
	ln -s /usr/bin/python3.6 /usr/bin/python
	sudo ln -s /usr/bin/python3.6 /usr/bin/python
else
	echo "Python 3 does not exist!!!"
	exit -1
fi

#apt-get autoremove -y libopencv-dev

make clean
make -j64
make py
#make test -j64
#make runtest -j64
