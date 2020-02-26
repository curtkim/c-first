## 개요
https://github.com/BVLC/caffe/tree/master/examples/cpp_classification

## Howto

    cd cmake-build-debug
    conan install .. --build missing
    cmake ..
    make 
    cd ..
    
    cmake-build-debug/bin/classify \
        models/bvlc_reference_caffenet/deploy.prototxt \
        models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
        data/ilsvrc12/imagenet_mean.binaryproto \
        data/ilsvrc12/synset_words.txt \
        examples/images/cat.jpg
        
## conan
https://github.com/bincrafters/conan-caffe/blob/testing/1.0/conanfile.py 