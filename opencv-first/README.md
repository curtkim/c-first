
    ldd sobel
        linux-vdso.so.1 (0x00007fff03f36000)
        libopencv_imgcodecs.so.4.1 => /home/curt/.conan/data/opencv/4.1.1/conan/stable/package/dd9dc6d61a9c9ce1de40485b2e367d5d584b2cfe/lib/libopencv_imgcodecs.so.4.1 (0x00007fd22661d000)
        libopencv_imgproc.so.4.1 => /home/curt/.conan/data/opencv/4.1.1/conan/stable/package/dd9dc6d61a9c9ce1de40485b2e367d5d584b2cfe/lib/libopencv_imgproc.so.4.1 (0x00007fd225da7000)
        libopencv_core.so.4.1 => /home/curt/.conan/data/opencv/4.1.1/conan/stable/package/dd9dc6d61a9c9ce1de40485b2e367d5d584b2cfe/lib/libopencv_core.so.4.1 (0x00007fd22575e000)
        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fd22537e000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fd225166000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fd224d75000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fd2249d7000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fd2247b8000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fd2245b4000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fd2243ac000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fd226ff8000)


## reference
https://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html