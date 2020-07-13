## Reference
https://blogs.oracle.com/linux/an-introduction-to-the-io_uring-asynchronous-io-framework

## HOWTO

    vagrant up
    vagrant ssh
    cd /data
    gcc -Wall -O2 -D_GNU_SOURCE -o uring-test uring-test.c -luring
    gcc -Wall -O2 -D_GNU_SOURCE -o uring-cp uring-cp.c -luring