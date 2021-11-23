docker build . -t cuda-conan-package
docker run --gpus all -v $(pwd):/data -it cuda-conan-package bash


# omega remote에 upload

cd /data/torch_prebuilt/
conan create . 1.7.1@curt/prebuilt -o torch:cuda=10.1
conan upload torch/1.7.1@curt/prebuilt -r omega --all

cd /data/torchvision/
conan create . curt/testing -o torchvision:with_cuda=True -o torch:cuda=10.1
conan upload torchvision/0.8.2@curt/testing -r omega --all
