#include "01_all.hpp"
#include <iostream>
#include <atomic>

#include "track.hpp"

#include "thread_pool_executor.hpp"

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

namespace {
    const int QUEUE_SIZE = 20;


    struct Frame {
        ring_span<std::tuple<Header, long>> lidar1;
        ring_span<std::tuple<Header, int>> camera1;
        ring_span<std::tuple<Header, int>> camera2;
        ring_span<std::tuple<Header, int>> gps1;
    };

    struct Timeline {
        Track<long> lidar1 = {QUEUE_SIZE};
        Track<int> camera1 = {QUEUE_SIZE};
        Track<int> camera2 = {QUEUE_SIZE};
        Track<int> gps1 = {QUEUE_SIZE};

        /*
        auto get_tracks() {
          return std::forward_as_tuple(lidar1, camera1, camera2);
        }
        */

        Frame frame() {
            return Frame{
                    lidar1.span(),
                    camera1.span(),
                    camera2.span(),
                    gps1.span()
            };
        }

        Frame frame(const Frame& prev) {
            return Frame{
                    lidar1.span(prev.lidar1),
                    camera1.span(prev.camera1),
                    camera2.span(prev.camera2),
                    gps1.span(prev.gps1)
            };
        }

        void release(Frame &frame) {
            for (auto &item : frame.lidar1)
                lidar1.dequeue();
            for (auto &item : frame.camera1)
                camera1.dequeue();
            for (auto &item : frame.camera2)
                camera2.dequeue();
            for (auto &item : frame.gps1)
                gps1.dequeue();
        }
    };

}

bool processing = false;
bool callback = false;

void log_timeline(const Timeline &timeline) {
    spdlog::info("timline lidar: {}~{}, gps: {}~{}",
                 timeline.lidar1.front, timeline.lidar1.rear,
                 timeline.gps1.front, timeline.gps1.rear);
}

void log_frame(spdlog::level::level_enum level, const Frame &frame, std::string_view msg) {
    if (frame.lidar1.size() > 0) {
        spdlog::log(level, "{} --- frame:  {} {} {} {}({}~{} {}~{} {}~{} {}~{})",
                     msg,
                     frame.lidar1.size(),
                     frame.camera1.size(),
                     frame.camera2.size(),
                     frame.gps1.size(),
                     std::get<0>(frame.lidar1.front()).seq,
                     std::get<0>(frame.lidar1.back()).seq,
                     std::get<0>(frame.camera1.front()).seq,
                     std::get<0>(frame.camera1.back()).seq,
                     std::get<0>(frame.camera2.front()).seq,
                     std::get<0>(frame.camera2.back()).seq,
                     std::get<0>(frame.gps1.front()).seq,
                     std::get<0>(frame.gps1.back()).seq);
    } else {
        spdlog::log(level,"{} --- frame not available");
    }
}

void process(const Frame &frame, int efd) {
    nvtxRangePush(__FUNCTION__);
    log_frame(spdlog::level::info, frame, "process");
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    const uint64_t value = 1;
    int ret = write(efd, &value, sizeof(uint64_t));
    if (ret != 8)
        handle_error("[producer] failed to write eventfd");
    spdlog::info("processed");
    nvtxRangePop();
}

void visualize(std::atomic_ref<Frame> frame_ref) {
    nvtxNameOsThread(syscall(SYS_gettid), "VIZ Thread");
    while (1) {
        nvtxRangePush(__FUNCTION__);
        const Frame &frame = frame_ref.load();
        log_frame(spdlog::level::debug, frame, "visualize");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        nvtxRangePop();
    }
}

void read_from_stream(asio::posix::stream_descriptor &stream, asio::mutable_buffer &buffer) {
    async_read(stream, buffer, [&stream, &buffer](const std::error_code ec, std::size_t) {
        spdlog::info("read from eventfd");
        processing = false;
        callback = true;
        read_from_stream(stream, buffer);
    });
}

int main() {
    nvtxNameOsThread(syscall(SYS_gettid), "IO Thread");
    spdlog::set_pattern("[%H:%M:%S.%e] [thread %t] %v");
    spdlog::set_level(spdlog::level::info);

    Timeline timeline;

    // initial empty frame
    Frame frame = timeline.frame();
    Frame frame_v = timeline.frame();
    std::atomic_ref<Frame> frame_v_ref(frame_v);

    size_t pool_size = 1;
    size_t max_pool_size = 1;
    size_t max_queue_size = 1;
    std::chrono::seconds keep_alive_time = std::chrono::seconds(5);

    std::thread visualize_thread(visualize, frame_v_ref);
    ThreadPoolExecutor executor(pool_size, max_pool_size, keep_alive_time, max_queue_size);

    std::cout << "sizeof timeline " << sizeof(timeline) << std::endl;
    std::cout << "sizeof lidar circular queue " << sizeof(timeline.lidar1) << std::endl;
    std::cout << "sizeof camera circular queue " << sizeof(timeline.camera1) << std::endl;

    int efd;
    efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (efd == -1)
        handle_error("eventfd");

    asio::io_service io_service;
    asio::posix::stream_descriptor stream{io_service, efd};

    uint64_t value;
    asio::mutable_buffer buffer = asio::buffer(&value, sizeof(value));
    read_from_stream(stream, buffer);

    TimerContext *lidar1Timer = setInterval(io_service, [&timeline]() {
        spdlog::debug("lidar1");
        nvtxRangePush("lidar1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeline.lidar1.enqueue(1);
        nvtxRangePop();
    }, 100);

    TimerContext *camera1Timer = setInterval(io_service, [&timeline]() {
        spdlog::debug("camera1");
        nvtxRangePush("camera1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeline.camera1.enqueue(1);
        nvtxRangePop();
    }, 100);

    TimerContext *camera2Timer = setInterval(io_service, [&timeline]() {
        spdlog::debug("camera2");
        nvtxRangePush("camera2");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeline.camera2.enqueue(1);
        nvtxRangePop();
    }, 100);

    TimerContext *gps1Timer = setInterval(io_service, [&timeline]() {
        spdlog::debug("gps1");
        nvtxRangePush("gps1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeline.gps1.enqueue(1);
        nvtxRangePop();
    }, 30);

    //io_service.run();
    while (true) {
        io_service.run_one();

        /*
        if( timeline.lidar1.size() > 0 && timeline.camera1.size() > 0 && timeline.camera2.size() > 0){
          frame = timeline.frame();
          spdlog::info("frame begin === {} {} {} {}",
                       frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
          timeline.release(frame);
          spdlog::info("frame release --- {} {} {} {}",
                       frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
        }
        */

        if (!processing) {
            if (callback) {
                log_timeline(timeline);

                timeline.release(frame_v);
                frame_v = frame;
                frame_v_ref.store(frame_v);

                spdlog::info("frame {}~{} frame_v {}~{}",
                             std::get<0>(frame.lidar1.front()).seq, std::get<0>(frame.lidar1.back()).seq,
                             std::get<0>(frame_v.lidar1.front()).seq, std::get<0>(frame_v.lidar1.back()).seq
                );
                spdlog::info("frame end ----------");

                callback = false;
            }

            if (timeline.lidar1.size() > 0 && timeline.camera1.size() > 0 && timeline.camera2.size() > 0) {
                frame = timeline.frame(frame_v);
                spdlog::info("frame begin ==========");
                spdlog::info("frame {}~{} frame_v {}~{}",
                             std::get<0>(frame.lidar1.front()).seq, std::get<0>(frame.lidar1.back()).seq,
                             std::get<0>(frame_v.lidar1.front()).seq, std::get<0>(frame_v.lidar1.back()).seq
                );
                processing = true;

                executor.submit([&frame, &efd]() {
                    process(frame, efd);
                });
            }
        }
    }

    visualize_thread.join();

    delete lidar1Timer;
    delete camera1Timer;
    delete camera1Timer;
    delete gps1Timer;

}