cmake_minimum_required (VERSION 3.16)

project(OpenMP LANGUAGES CXX Fortran)

include(cmake/CPM.cmake)

find_package(MPI REQUIRED)

# include(CTest)

# CPMAddPackage(
#   NAME GTest
#   GITHUB_REPOSITORY google/googletest
#   GIT_TAG release-1.11.0
#   EXCLUDE_FROM_ALL ON
# )
# include(GoogleTest)

CPMAddPackage(
  NAME xtl
  GITHUB_REPOSITORY xtensor-stack/xtl
  GIT_TAG 0.7.5
  VERSION 0.7.5
)

CPMAddPackage(
  NAME xtensor
  GITHUB_REPOSITORY xtensor-stack/xtensor
  GIT_TAG 0.24.5
  VERSION 0.24.5
)

add_executable(test_mpi test_mpi.cxx)
target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)

add_executable(mpi_message mpi_message.cxx)
target_link_libraries(mpi_message PRIVATE MPI::MPI_CXX)

add_executable(mpi_dl_test mpi_dl_test.cxx)
target_link_libraries(mpi_dl_test PRIVATE MPI::MPI_CXX)

add_executable(test_integration test_integration.cxx)
target_link_libraries(test_integration PRIVATE MPI::MPI_CXX)

add_executable(test_derivative test_derivative.cxx derivative.cxx)
target_link_libraries(test_derivative PRIVATE MPI::MPI_CXX xtensor)

add_executable(wave_equation wave_equation.cxx derivative.cxx)
target_link_libraries(wave_equation PRIVATE MPI::MPI_CXX xtensor)

add_executable(kdv kdv.cxx)
target_link_libraries(kdv PRIVATE MPI::MPI_CXX xtensor)

add_executable(heat_eqn heat_eqn.cxx)
target_link_libraries(heat_eqn PRIVATE MPI::MPI_CXX xtensor)

add_subdirectory(omp)
