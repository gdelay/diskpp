option(BUILD_APP_UC "Build unique continuation application" ON)
if (BUILD_APP_UC)

  configure_file(test2d_1.geo test2d_1.geo COPYONLY)
  configure_file(test2d_2.geo test2d_2.geo COPYONLY)
  configure_file(test2d_3.geo test2d_3.geo COPYONLY)
  configure_file(test2d_4.geo test2d_4.geo COPYONLY)

  add_executable(heat_UC heat_UC.cpp)
  target_link_libraries(heat_UC ${LINK_LIBS})
endif ()
