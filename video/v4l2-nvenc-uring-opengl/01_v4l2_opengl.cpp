#include <iostream>
#include <string>
#include <thread>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>

#include <nvToolsExt.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "utils/capture_utils.hpp"
#include "utils/opengl_utils.hpp"
#include "utils/stopwatch.hpp"



#define CLEAR(x) memset(&(x), 0, sizeof(x))

static int WIDTH = 640;
static int HEIGHT = 480;
static int FIXEL_FORMAT = V4L2_PIX_FMT_YUYV;

template<typename Callable>
void nvtxRange(char* range_name, Callable body) {
    nvtxRangePush(range_name);
    body();
    nvtxRangePop();
}


int main() {

    using namespace std::chrono;


    nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
    std::cout << "main thread: " << std::this_thread::get_id() << std::endl;
    printf("sizeof(v4l2_buffer)=%d\n", sizeof(v4l2_buffer));    // 881

    // init v4l2
    DeviceContext deviceInfo;
    deviceInfo.dev_name = "/dev/video2";

    open_device(deviceInfo);
    init_device(deviceInfo, WIDTH, HEIGHT, FIXEL_FORMAT);

    int epfd = epoll_create(1);
    struct epoll_event ev;
    ev.data.fd = deviceInfo.fd;
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);

    start_capturing(deviceInfo);


    // init opengl
    GLFWwindow *window = make_window(WIDTH, HEIGHT);
    auto[VAO, VBO, EBO] = viz::bg::load_model();
    Shader bgShader(viz::bg::VERTEX_SHADER_SOURCE, viz::bg::FRAGMENT_SHADER_SOURCE);


    int frame = 0;
    while (!glfwWindowShouldClose(window))
    {
        auto start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        frame++;

        // input
        // -----
        processInput(window);

        // get from queue
        StopWatch watch;
        struct epoll_event events[1];
        struct v4l2_buffer buf;

        while(1){
            watch.reset();
            int nfds = epoll_wait(epfd, events, 1, 10*1000);
            printf("%d epoll elapsed_time = %ld nfds=%d\n", frame, watch.get_elapsed_time(), nfds);
            if (nfds <= 0)
                continue;

            if (deque_buffer(deviceInfo, buf))
                break;
        }

        cv::Mat B;
        nvtxRange("cvtColor", [&deviceInfo, &buf, &B](){
            cv::Mat A(HEIGHT, WIDTH, CV_8UC2, deviceInfo.buffers[buf.index].start);
            // yvyu -> rgb
            cvtColor(A, B, CV_YUV2RGB_YVYU); // YUV2RGB인데 BGR로 변환되는 것 같다.
        });

        queue_buffer(deviceInfo, buf);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        bgShader.use();
        glBindVertexArray(VAO);
        nvtxRange("viz", [&B](){
            unsigned int texture = viz::bg::load_texture(WIDTH, HEIGHT, GL_BGR, B.data);
            glBindTexture(GL_TEXTURE_2D, texture);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glDeleteTextures(1, &texture);
        });

        glfwSwapBuffers(window);
        glfwPollEvents();

        auto end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        long diff = (end_time - start_time).count();
        printf("%d frame elapsed_time = %ld \n", frame, diff);
    }

    viz::bg::delete_model(VAO, VBO, EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();


    epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
    stop_capturing(deviceInfo);
    uninit_device(deviceInfo);
    close_device(deviceInfo);

    std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}