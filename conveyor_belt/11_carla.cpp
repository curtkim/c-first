#include "11_all.hpp"

#include "track.hpp"
#include "ring_span.hpp"

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)


using namespace std::chrono_literals;

struct DeviceBuffer {
    void* ptr;
};


// anonymous namespace
namespace {
    const int QUEUE_SIZE = 20;


    struct Frame {
        ring_span<std::tuple<Header, DeviceBuffer>> camera1;
    };

    struct Timeline {
        Track<DeviceBuffer> camera1 = {QUEUE_SIZE};

        Frame frame(){
            return Frame{
                    camera1.span(),
            };
        }

        Frame frame(const Frame& prev) {
            return Frame{
                    camera1.span(prev.camera1),
            };
        }

        void release(const Frame& frame) {
            for(auto& [header, item] : frame.camera1) {
                camera1.dequeue();

                spdlog::info("cudaFree {}", item.ptr);
                cudaError ce = cudaFree(item.ptr);
                if(ce != 0)
                    handle_error("cudaFree");
            }
        }
    };

}

bool processing = false;
bool callback = false;

void log_timeline(const Timeline& timeline){
    spdlog::info("timline camera: {}~{}",
                 timeline.camera1.front, timeline.camera1.rear);
}

void log_frame(const Frame& frame, std::string_view msg) {
    if(frame.camera1.size() > 0) {
        spdlog::info("{} --- frame:  {}({}~{})",
                     msg,
                     frame.camera1.size(),
                     std::get<0>(frame.camera1.front()).seq,
                     std::get<0>(frame.camera1.back()).seq);
    }
    else{
        spdlog::info("{} --- frame not available", msg);
    }
}


void process(const Frame& frame, int efd){
    nvtxRangePush(__FUNCTION__);
    log_frame(frame, "process");
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    const uint64_t value = 1;
    int ret = write(efd, &value, sizeof(uint64_t));
    if (ret != 8)
        handle_error("[producer] failed to write eventfd");
    spdlog::info("processed");
    nvtxRangePop();
}

void visualize(std::atomic_ref<Frame> frame_ref){
    nvtxNameOsThread(syscall(SYS_gettid), "VIZ Thread");
    while(1){
        nvtxRangePush(__FUNCTION__);
        const Frame& frame = frame_ref.load();
        log_frame(frame, "visualize");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        nvtxRangePop();
    }
}


void read_from_stream(asio::posix::stream_descriptor& stream, asio::mutable_buffer& buffer) {
    async_read(stream, buffer, [&stream, &buffer](const std::error_code ec, std::size_t) {
        spdlog::info("read from eventfd");
        processing = false;
        callback = true;
        read_from_stream(stream, buffer);
    });
}


int main() {
    namespace cc = carla::client;
    namespace cg = carla::geom;
    namespace csd = carla::sensor::data;

    cudaSetDevice(1);

    nvtxNameOsThread(syscall(SYS_gettid), "IO Thread");
    spdlog::set_pattern("[%H:%M:%S.%e] [thread %t] %v");


    Timeline timeline;
    // initial empty frame
    Frame frame = timeline.frame();
    Frame frame_v = timeline.frame();
    std::atomic_ref<Frame> frame_v_ref(frame_v);

    moodycamel::ReaderWriterQueue<DeviceBuffer> q(100);

    size_t pool_size = 1;
    size_t max_pool_size = 1;
    size_t max_queue_size = 1;
    std::chrono::seconds keep_alive_time = std::chrono::seconds(5);
    ThreadPoolExecutor executor(pool_size, max_pool_size, keep_alive_time, max_queue_size);

    std::thread visualize_thread(visualize, frame_v_ref);


    int efd;
    efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (efd == -1)
        handle_error("eventfd");

    asio::io_service io_service;
    asio::posix::stream_descriptor stream{io_service, efd};

    uint64_t value;
    asio::mutable_buffer buffer = asio::buffer(&value, sizeof(value));
    read_from_stream(stream, buffer);


    auto client = cc::Client("localhost", 2000, 1);
    client.SetTimeout(10s);

    auto world = client.GetWorld();
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");
    auto blueprint = (*vehicles)[0];

    auto transform = cg::Transform{
            cg::Location{-36.6, -194.9, 0.27},
            cg::Rotation{0, 1.4395, 0}};
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(world.SpawnActor(blueprint, transform));
    std::cout << "Spawned " << vehicle->GetDisplayId() << '\n';

    auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
    assert(camera_bp != nullptr);
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.2");

    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
            cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
            cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

    // Register a callback to save images to disk.
    camera->Listen([&q](auto data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        size_t size = image->GetWidth() * image->GetHeight() * 3;

        DeviceBuffer buffer;
        cudaError ce = cudaMalloc((void **)&(buffer.ptr), size);
        if( ce != 0){
            handle_error("cudaMalloc");
        }
        spdlog::info("listen cudaMalloc size={} ptr={}", size, buffer.ptr);

        bool succeeded = q.enqueue(buffer);
        if(!succeeded){
            handle_error("q.enqueue");
        }
    });

    TimerContext* timer = setInterval(io_service, [](){
        //std::cout << "timer : " << std::this_thread::get_id() << std::endl;
    }, 10);


    while(true){
        io_service.run_one();

        DeviceBuffer buffer;
        bool success = true;
        while(success){
            success = q.try_dequeue(buffer);
            if( success)
                timeline.camera1.enqueue(buffer);
        }

        if( !processing ){
            if( callback){
                log_timeline(timeline);

                spdlog::info("frame release --- camera.seq={}~{}",
                             std::get<0>(frame_v.camera1.front()).seq,
                             std::get<0>(frame_v.camera1.back()).seq);
                timeline.release(frame_v);
                frame_v = frame;
                frame_v_ref.store(frame_v);

                callback = false;
            }

            if( timeline.camera1.size() > 0){
                frame = timeline.frame(frame_v);
                spdlog::info("frame begin === camera.seq={}~{}",
                             std::get<0>(frame.camera1.front()).seq,
                             std::get<0>(frame.camera1.back()).seq);
                processing = true;

                executor.submit([&frame, &efd](){
                    process(frame, efd);
                });
            }
        }
    }

    visualize_thread.join();

    delete timer;
}