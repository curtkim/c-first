clang++ -std=c++2a -fmodules-ts --precompile math.cppm -o math.pcm                   
clang++ -std=c++2a -fmodules-ts -c math.pcm -o math.o                                
clang++ -std=c++2a -fmodules-ts -fprebuilt-module-path=. math.o main.cpp -o math     

