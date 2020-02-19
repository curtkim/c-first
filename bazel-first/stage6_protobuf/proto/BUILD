load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "sensor_cc_proto",
    deps = [
        ":sensor_proto",
    ],
)

proto_library(
    name = "sensor_proto",
    srcs = [
        "sensor.proto",
    ],
)