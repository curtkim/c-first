cmake -S step1 -B step1_build
(cd step1_build && make && ./Tutorial)
echo '***'
cmake -S step2 -B step2_build
(cd step1_build && make && ./Tutorial)
echo '***'
cmake -S step3 -B step3_build
(cd step3_build && make && make install && ctest)