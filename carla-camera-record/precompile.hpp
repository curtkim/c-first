#pragma once

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/sensor/data/Image.h>

#include <nppi.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda.h>
#include "nvEncodeAPI.h"
#include "NvEncoder/NvEncoderCuda.h"
#include <nvToolsExt.h>

#include <sys/syscall.h>

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
