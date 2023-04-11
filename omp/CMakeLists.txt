
find_package(OpenMP REQUIRED)

add_executable(test_openmp test_openmp.cxx)
target_link_libraries(test_openmp OpenMP::OpenMP_CXX)
