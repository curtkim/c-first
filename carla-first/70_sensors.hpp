#pragma once

#include "carla_common.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;


std::map<std::string, cg::Transform> CAMERA_TOPIC_TRANSFORM_MAP{
  {"/camera/0", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/1", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/2", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/3", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/4", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/5", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/6", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/7", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
  {"/camera/8", cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},
    cg::Rotation{-15.0f, 0.0f, 0.0f}}},
};
