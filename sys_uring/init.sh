apt-get update
apt-get install -y make
apt-get install -y gcc g++

git clone https://github.com/axboe/liburing.git
cd liburing
./configure
make install

#wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh
#chmod +x ubuntu-mainline-kernel.sh
#pwd
#./ubuntu-mainline-kernel.sh -i v5.8.18
