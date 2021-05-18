## Step
- 


## HOWTO

    nsys profile -o _01_cam1_thread1_cvcolor.qdstrm --force-overwrite=true cmake-build-debug/bin/01_cam1_thread1_cvcolor
    nsys profile -o _02_cam1_thread1_npp.qdstrm --force-overwrite=true cmake-build-debug/bin/02_cam1_thread1_npp
    nsys profile -o _03_cam3_thread1_npp.qdstrm --force-overwrite=true cmake-build-debug/bin/03_cam3_thread1_npp

    nsys-ui 
