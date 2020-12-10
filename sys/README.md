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
    
    
## Reference
- Linux AsyncIO : https://blog.cloudflare.com/io_submit-the-epoll-alternative-youve-never-heard-about/
### eventfd
- https://www.yangyang.cloud/blog/2018/11/09/worker-pool-with-eventfd/
- https://github.com/Pro-YY/eventfd_examples


## 26_eventfd_worker.c 골격

    epfd = epoll_create1(EPOLL_CLOEXEC);

    # producer
    int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);

    struct epoll_event event;
    event.data.fd = efd;
    event.events = EPOLLIN | EPOLLET;
    ret = epoll_ctl(epfd, EPOLL_CTL_ADD, efd, &event);
    ret = write(efd, &(uint64_t) {1}, sizeof(uint64_t));


    # consumer
    nfds = epoll_wait(epfd, events, MAX_EVENTS_SIZE, 1000); 
    for(event : events)
      ret = read(event.data.fd, &v, sizeof(v));
      close(event.data.fd);


