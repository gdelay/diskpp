set(LINK_LIBS diskpp)

## Required C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(membranes src/membranes.cpp)
target_link_libraries(membranes ${LINK_LIBS})
install(TARGETS membranes RUNTIME DESTINATION bin)
