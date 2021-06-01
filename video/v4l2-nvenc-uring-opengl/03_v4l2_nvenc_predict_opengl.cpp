#include <iostream>
#include <string>
#include <thread>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/epoll.h>

#include <cuda.h>
#include "nvEncodeAPI.h"
#include "NvEncoder/NvEncoderCuda.h"
#include <nvToolsExt.h>


#include <npp.h>

#include "utils/capture_utils.hpp"
#include "utils/opengl_utils.hpp"
#include "utils/stopwatch.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/nvtx_utils.hpp"

#include "detr.hpp"

#include <cuda_gl_interop.h>


#define CLEAR(x) memset(&(x), 0, sizeof(x))


const int WIDTH = 640;
const int HEIGHT = 480;

const int FIXEL_FORMAT = V4L2_PIX_FMT_YUYV;


NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
GUID codecGuid = NV_ENC_CODEC_H264_GUID;
GUID presetGuid = NV_ENC_PRESET_P3_GUID;
NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;


int main() {

    using namespace std::chrono;


    nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
    std::cout << "main thread: " << std::this_thread::get_id() << std::endl;
    printf("sizeof(v4l2_buffer)=%d\n", sizeof(v4l2_buffer));    // 881

    // 0번 gpu는 carla가 쓰고있다.
    auto torch_device = torch::Device(torch::kCUDA, 1);
    torch::jit::script::Module detr_model = detr::load_model("../../wrapped_detr_resnet50.pt", torch_device);


    // init encoder
    int iGpu = 0;
    ck(cuInit(0));
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    std::ofstream out("02_v4l2_nvenc_opengl.h264", std::ios::out | std::ios::binary);

    NvEncoderCuda enc(cuContext, WIDTH, HEIGHT, eFormat);
    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        enc.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        enc.CreateEncoder(&initializeParams);
    }


    // init v4l2
    DeviceContext deviceInfo;
    deviceInfo.dev_name = "/dev/video0";

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

    unsigned int texture0;// = load_texture("00000_camera0.png", true, GL_RGBA);
    {
        glGenTextures(1, &texture0);
        glBindTexture(GL_TEXTURE_2D, texture0);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }
    // PIXEL_UNPACK_BUFFER bind
    unsigned int image_pixel_buffer_;
    {
        glGenBuffers(1, &image_pixel_buffer_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH*HEIGHT*3, 0, GL_STATIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    Shader bgShader(viz::bg::VERTEX_SHADER_SOURCE, viz::bg::FRAGMENT_SHADER_SOURCE);


    int frame = 0;
    Npp8u *pSrc, *pDst, *pDstYUV420;
    cudaMalloc(&pSrc, HEIGHT*WIDTH*2);
    cudaMalloc(&pDst, HEIGHT*WIDTH*3);
    cudaMalloc(&pDstYUV420, HEIGHT*WIDTH*3/2);
    std::vector<std::vector<uint8_t>> vPacket;

    cudaGraphicsResource *cuda_gl_resource;
    ck(cudaGraphicsGLRegisterBuffer(&cuda_gl_resource, image_pixel_buffer_, cudaGraphicsMapFlagsNone));

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

        nvtxRange("color_convert", [&](){
            cudaMemcpy(pSrc, deviceInfo.buffers[buf.index].start, HEIGHT*WIDTH * 2, cudaMemcpyHostToDevice);

            // pDst로 RGB 변환
            NppiSize oSizeROI{WIDTH, HEIGHT};
            NppStatus res = nppiYUV422ToRGB_8u_C2C3R(pSrc, WIDTH * 2, pDst, WIDTH*3, oSizeROI);
            if (res != 0) {
                printf("oops %d\n", (int) res);
                std::exit(1);
            }

            // pDstYUV420으로 YUV420 변환
            Npp8u *pDst3[3] = {pDstYUV420, pDstYUV420 + (WIDTH * HEIGHT), pDstYUV420 + (WIDTH * HEIGHT) * 5 / 4};
            int rDstStep[3] = {WIDTH * sizeof(Npp8u), (WIDTH / 2) * sizeof(Npp8u), (WIDTH / 2) * sizeof(Npp8u)};

            res = nppiRGBToYUV420_8u_C3P3R(pDst, WIDTH * 3, pDst3, rDstStep, oSizeROI);
            if (res != 0) {
                printf("oops %d\n", (int) res);
                std::exit(1);
            }
        });

        queue_buffer(deviceInfo, buf);


        nvtxRange("encoder_copy_frame", [&](){
            const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pDstYUV420, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                             (int) encoderInputFrame->pitch,
                                             enc.GetEncodeWidth(),
                                             enc.GetEncodeHeight(),
                                             CU_MEMORYTYPE_DEVICE,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes);
        });

        nvtxRange("encode", [&](){
            enc.EncodeFrame(vPacket);
        });

        nvtxRange("file_write", [&]() {
            for (std::vector<uint8_t> &packet : vPacket) {
                printf("%ld write\n", packet.size());
                // For each encoded packet
                out.write(reinterpret_cast<char *>(packet.data()), packet.size());
            }
        });

        {
            // pDst로 부터 copy한다.
            // scale하고 crop해서 1024, 800에 맞춘다.

//            at::Allocator *allocator = at::cuda::getCUDADeviceAllocator();
//
//            torch::DataPtr d_data = allocator->allocate(N * sizeof(float));
//            cudaMemcpy(d_data.get(), data, N * sizeof(float), cudaMemcpyHostToDevice);
//
//            auto options =
//                    torch::TensorOptions()
//                            .dtype(torch::kFloat32)
//                            .device(torch::kCUDA);
//
//            torch::Tensor cudaTest = torch::from_blob(d_data.get(), {N}, options);

        }

        auto bounding_boxes = detr::detect(detr_model, img);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        bgShader.use();
        glBindVertexArray(VAO);

        nvtxRange("copy_texture", [&](){
            uint8_t *raw_render_image_ptr;
            size_t n_bytes;

            ck(cudaGraphicsMapResources(1, &cuda_gl_resource));
            ck(cudaGraphicsResourceGetMappedPointer((void **) &raw_render_image_ptr, &n_bytes, cuda_gl_resource));
            //std::cout << n_bytes << " " << width * height * 4 << std::endl;
            ck(cudaMemcpy(raw_render_image_ptr, pDst, HEIGHT*WIDTH*3, cudaMemcpyDeviceToDevice));
            ck(cudaGraphicsUnmapResources(1, &cuda_gl_resource));
        });

        nvtxRange("draw", [&](){
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture0);

            // 아래 3줄이 필요함.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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