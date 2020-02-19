load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
    visibility = ["//main:__pkg__"],
)

cc_library(
    name = "hello-math",
    srcs = ["hello-math.cc"],
    hdrs = ["hello-math.h"],
    visibility = ["//main:__pkg__"],
)

cc_test(
    name = "hello-math-test",
    srcs = glob(["**/*-test.cc"]),
    deps = [":hello-math",
           "@gtest//:gtest_main"],
)