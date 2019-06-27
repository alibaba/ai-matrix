#!/bin/bash

apt-get update

ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.1.0 /usr/lib/x86_64-linux-gnu/libhdf5.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

#apt-get install -y libboost-all-dev  gfortran libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev liblmdb-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libhdf5-serial-dev libleveldb-dev libopencv-dev libsnappy-dev
apt-get install -y libboost-all-dev  gfortran libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev liblmdb-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev libopencv-dev libsnappy-dev

#apt-get install -y python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-h5py python-protobuf python-leveldb python-networkx python-nose python-pandas python-gflags cython ipython python-yaml 
apt-get install -y python-numpy python-scipy python-matplotlib python-sklearn python-skimage python-h5py python-leveldb python-networkx python-nose python-pandas python-gflags cython ipython python-yaml 

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

pip3 install scikit-image
#apt-get autoremove -y libopencv-dev

if [ ! -f /usr/local/lib/libprotobuf.so.9 ]; then
	wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
	tar -zxvf protobuf-2.6.1.tar.gz
	apt-get install build-essential
	cd protobuf-2.6.1
	./configure
	make -j64
	make check -j64
	make install -j64
	cd ..
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

pip3 install easydict
pip3 install pyyaml

#protoc src/caffe/proto/caffe.proto --cpp_out=.
#mkdir include/caffe/proto
#mv src/caffe/proto/caffe.pb.h include/caffe/proto

cd lib
make
cd ..

cd caffe-fast-rcnn
make clean
make -j64
make pycaffe
cd ..

./data/scripts/fetch_faster_rcnn_models.sh
#make test -j64
#make runtest -j64
