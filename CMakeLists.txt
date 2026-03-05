cmake_minimum_required(VERSION 3.24)

project(now_where_did_i_put_that)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(where_test tests/where_test.cpp src/where.cpp)
add_executable(main main.cpp src/where.cpp)

target_link_libraries(where_test PRIVATE Catch2::Catch2WithMain)
