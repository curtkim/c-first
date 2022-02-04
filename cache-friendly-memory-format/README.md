
    from https://pytorch.org/blog/tensor-memory-format-matters/

    g++ -O2 row_first_column_first.cpp -DRUN_LOOP1 -DRUN_LOOP2

    sudo apt install valgrind
    g++ -O2 row_first_column_first.cpp -DRUN_LOOP1
	valgrind --tool=cachegrind ./a.out

    g++ -O2 row_first_column_first.cpp -DRUN_LOOP2
	valgrind --tool=cachegrind ./a.out
