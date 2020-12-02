## Reference
- https://blogs.oracle.com/linux/an-introduction-to-the-io_uring-asynchronous-io-framework
- https://unixism.net/loti/tutorial/cat_liburing.html

## HOWTO

    vagrant up
    vagrant provision
    vagrant ssh
    cd /data
    cmake -S . -B build
    cd build && make
    
    
## cat_uring

    io_uring_queue_init()
        io_uring_get_sqe()
            io_uring_prep_readv()
            io_uring_sqe_set_data()  -- set_data
        io_uring_submit()
        io_uring_wait_cqe()
            io_uring_cqe_get_data()  -- get_data
    io_uring_queue_exit()

## uring-cp
    
    