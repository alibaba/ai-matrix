CUR=$(pwd)
echo $CUR
mkdir Model
cd ./Model
wget https://github.com/opencv/opencv/archive/3.4.0.zip
apt-get install cmake -y
unzip 3.4.0.zip
cd opencv-3.4.0
mkdir build && cd build
cmake -D WITH_NVCUVID=OFF CMAKE_INSTALL_PREFIX=$CUR/Model/cv3 ..
make -j$(nproc) && make install
cd $CUR
make
cp giexec ./bin
