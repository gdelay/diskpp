option(BUILD_APP_UC "Build unique continuation application" ON)
if (BUILD_APP_UC)

    add_executable(heat_UC heat_UC.cpp)
    target_link_libraries(heat_UC ${LINK_LIBS})
endif ()
