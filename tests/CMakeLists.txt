# Test suite CMake configuration
add_executable(test_linear_regression test_linear_regression.cpp)
target_link_libraries(test_linear_regression PRIVATE sklcpp Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_linear_regression)
