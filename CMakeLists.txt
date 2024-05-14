cmake_minimum_required(VERSION 3.14)
project(distributed_training)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  tensor_test
  tensor.cpp
  test/tensor_test.cpp
)

add_executable(
  ops_test
  ops.cpp
  tensor.cpp
  test/ops_test.cpp
)

target_link_libraries(
  tensor_test
  GTest::gtest_main
)
target_link_libraries(
  ops_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_test)