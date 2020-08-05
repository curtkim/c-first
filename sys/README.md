## test_read performance
    
    fallocate -l 4G number.bin
    
    test_read_ifstream
    10908
    10969
    10913
    10905
    10920
    Average exec time: 10923
    
    test_read_mmap
    9713
    9621
    9740
    9669
    9709
    Average exec time: 9690    