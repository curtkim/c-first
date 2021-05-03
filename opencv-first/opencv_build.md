https://webnautes.tistory.com/1479


    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_C_COMPILER=/usr/bin/gcc-9 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PACKAGE=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_CUFFT=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_IPP=OFF \
    -D WITH_V4L=ON \
    -D WITH_1394=OFF \
    -D WITH_GTK=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_EIGEN=ON \
    -D WITH_FFMPEG=OFF \
    -D WITH_GSTREAMER=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=OFF \
    -D OPENCV_SKIP_PYTHON_LOADER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.2/modules \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D CUDA_ARCH_BIN=6.1 \
    -D CUDA_ARCH_PTX=6.1 \
    ..



/usr/bin/c++  -pthread  -g  CMakeFiles/90_gpumat.dir/90_gpumat.cpp.o -o bin/90_gpumat   
-L/home/curt/.conan/data/opencv/4.5.2/_/_/package/4363a0a0beba9ffd21d0d9b16aac97a2ceb70697/lib  
-L/home/curt/.conan/data/jasper/2.0.25/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib  
-L/home/curt/.conan/data/libpng/1.6.37/_/_/package/f99afdbf2a1cc98ba2029817b35103455b6a9b77/lib  
-L/home/curt/.conan/data/openexr/2.5.5/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib  
-L/home/curt/.conan/data/libtiff/4.2.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib  
-L/home/curt/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib  
-L/home/curt/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libdeflate/1.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib  
-L/home/curt/.conan/data/zstd/1.4.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libwebp/1.1.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib  
-L/usr/local/cuda/lib64  
-Wl,
-rpath,/home/curt/.conan/data/opencv/4.5.2/_/_/package/4363a0a0beba9ffd21d0d9b16aac97a2ceb70697/lib:
/home/curt/.conan/data/jasper/2.0.25/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib:
/home/curt/.conan/data/libpng/1.6.37/_/_/package/f99afdbf2a1cc98ba2029817b35103455b6a9b77/lib:
/home/curt/.conan/data/openexr/2.5.5/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib:
/home/curt/.conan/data/libtiff/4.2.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib:
/home/curt/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib:
/home/curt/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/libdeflate/1.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib:
/home/curt/.conan/data/zstd/1.4.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/libwebp/1.1.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib:
/usr/local/cuda/lib64 
-lopencv_stitching 
-lopencv_intensity_transform 
-lopencv_quality 
-lopencv_reg 
-lopencv_surface_matching -lopencv_xphoto -lopencv_alphamat -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash 
-lopencv_line_descriptor -lopencv_saliency -lopencv_rapid -lopencv_rgbd -lopencv_structured_light 
-lopencv_phase_unwrapping -lopencv_videostab -lopencv_xfeatures2d -lopencv_shape -lopencv_xobjdetect 
-lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_highgui 
-lopencv_face -lopencv_photo -lopencv_superres -lopencv_videoio -lopencv_optflow -lopencv_ximgproc 
-lopencv_stereo -lopencv_tracking -lopencv_plot -lopencv_datasets -lopencv_ml -lopencv_imgcodecs 
-lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudacodec -lopencv_cudaimgproc -lopencv_cudalegacy 
-lopencv_video -lopencv_cudaobjdetect -lopencv_objdetect -lopencv_cudaoptflow -lopencv_cudastereo 
-lopencv_calib3d -lopencv_flann -lopencv_cudafeatures2d -lopencv_cudafilters -lopencv_cudawarping 
-lopencv_imgproc -lopencv_core -lopencv_cudev 
-ljasper 
-lpng16 
-lIlmImf-2_5 -lIlmImfUtil-2_5 -lIlmThread-2_5 -lImath-2_5 -lHalf-2_5 -lIexMath-2_5 -lIex-2_5 
-ltiffxx -ltiff -lquirc -lz -ljpeg -ldeflate -llzma -ljbig -lzstd 
-lwebpdecoder -lwebpdemux -lwebpmux -lwebp -ldl -lrt -lstdc++ 
-lgtk-x11-2.0 -lgdk-x11-2.0 -lpangocairo-1.0 -latk-1.0 
-lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0 -lgobject-2.0 -lglib-2.0 
-lharfbuzz -lfontconfig -lfreetype -lpthread -lm 
-lnvrtc -lcudart


/usr/bin/c++  -pthread  -g  CMakeFiles/50_epipolar_lines.dir/50_epipolar_lines.cpp.o -o bin/50_epipolar_lines   
-L/home/curt/.conan/data/opencv/4.5.2/_/_/package/4363a0a0beba9ffd21d0d9b16aac97a2ceb70697/lib  
-L/home/curt/.conan/data/glad/0.1.34/_/_/package/18346ee21bbc8282fbdd084747fe37b49b6f1517/lib  
-L/home/curt/.conan/data/glfw/3.3.4/_/_/package/6340505dafa41af473a127b95b9c69164d638b69/lib  
-L/home/curt/.conan/data/jasper/2.0.25/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib  
-L/home/curt/.conan/data/libpng/1.6.37/_/_/package/f99afdbf2a1cc98ba2029817b35103455b6a9b77/lib  
-L/home/curt/.conan/data/openexr/2.5.5/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib  
-L/home/curt/.conan/data/libtiff/4.2.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib  
-L/home/curt/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib  
-L/home/curt/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libdeflate/1.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib  
-L/home/curt/.conan/data/zstd/1.4.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib  
-L/home/curt/.conan/data/libwebp/1.1.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib  
-L/usr/local/cuda/lib64  
-Wl,
-rpath,/home/curt/.conan/data/opencv/4.5.2/_/_/package/4363a0a0beba9ffd21d0d9b16aac97a2ceb70697/lib:
/home/curt/.conan/data/glad/0.1.34/_/_/package/18346ee21bbc8282fbdd084747fe37b49b6f1517/lib:
/home/curt/.conan/data/glfw/3.3.4/_/_/package/6340505dafa41af473a127b95b9c69164d638b69/lib:
/home/curt/.conan/data/jasper/2.0.25/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib:
/home/curt/.conan/data/libpng/1.6.37/_/_/package/f99afdbf2a1cc98ba2029817b35103455b6a9b77/lib:
/home/curt/.conan/data/openexr/2.5.5/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib:
/home/curt/.conan/data/libtiff/4.2.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib:
/home/curt/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib:
/home/curt/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/libdeflate/1.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:
/home/curt/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib:
/home/curt/.conan/data/zstd/1.4.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib:/
home/curt/.conan/data/libwebp/1.1.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib:
/usr/local/cuda/lib64 

-lopencv_stitching 
-lopencv_intensity_transform 
-lopencv_quality 
-lopencv_reg 
-lopencv_surface_matching 
-lopencv_xphoto 
-lopencv_alphamat 
-lopencv_fuzzy 
-lopencv_hfs 
-lopencv_img_hash 
-lopencv_line_descriptor 
-lopencv_saliency 
-lopencv_rapid 
-lopencv_rgbd 
-lopencv_structured_light 
-lopencv_phase_unwrapping -lopencv_videostab -lopencv_xfeatures2d -lopencv_shape -lopencv_xobjdetect -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_photo -lopencv_superres -lopencv_videoio -lopencv_optflow -lopencv_ximgproc -lopencv_stereo -lopencv_tracking -lopencv_plot -lopencv_datasets -lopencv_ml -lopencv_imgcodecs -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudacodec -lopencv_cudaimgproc -lopencv_cudalegacy -lopencv_video -lopencv_cudaobjdetect -lopencv_objdetect -lopencv_cudaoptflow -lopencv_cudastereo -lopencv_calib3d 
-lopencv_flann 
-lopencv_cudafeatures2d 
-lopencv_cudafilters 
-lopencv_cudawarping 
-lopencv_imgproc 
-lopencv_core 
-lopencv_cudev 
-lglad -lglfw3 -ljasper -lpng16 -lIlmImf-2_5 -lIlmImfUtil-2_5 -lIlmThread-2_5 -lImath-2_5 -lHalf-2_5 -lIexMath-2_5 -lIex-2_5 -ltiffxx -ltiff -lquirc -lz 
-ljpeg -ldeflate -llzma -ljbig -lzstd -lwebpdecoder -lwebpdemux -lwebpmux -lwebp -ldl -lrt -lstdc++ -lgtk-x11-2.0 -lgdk-x11-2.0 -lpangocairo-1.0 -latk-1.0 
-lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0 -lgobject-2.0 -lglib-2.0 -lharfbuzz -lfontconfig -lfreetype 
-lGL -lX11 -lX11-xcb -lxcb -lfontenc -lICE -lSM -lXau -lXaw7 -lXt -lXcomposite -lXcursor -lXdamage -lXfixes -lXdmcp -lXext -lXft -lXi -lXinerama -lxkbfile 
-lXmu -lXmuu -lXpm -lXrandr -lXrender -lXRes -lXss -lXtst -lXv -lXvMC -lXxf86vm -lxcb-xkb -lxcb-icccm -lxcb-image -lxcb-shm -lxcb-keysyms -lxcb-randr 
-lxcb-render -lxcb-render-util -lxcb-shape -lxcb-sync -lxcb-xfixes -lxcb-xinerama -lxcb-util -lxcb-dri3 
-lpthread -lm -lnvrtc -lcudart 
