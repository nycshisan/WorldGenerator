cmake_minimum_required(VERSION 3.8)
project(WorldGenerator)

set(CMAKE_CXX_STANDARD 11)

include_directories(lib/rapidjson1.1.0 lib/sfml2.4.2/include)

link_directories(lib/sfml2.4.2/lib)
link_libraries(libsfml-window.dylib libsfml-system.dylib libsfml-audio.dylib libsfml-graphics.dylib libsfml-network.dylib)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(SOURCE_FILES src/main.cpp src/conf/conf.cpp src/conf/conf.h src/graphics/window.cpp src/graphics/window.h src/log.h)
add_executable(WorldGenerator ${SOURCE_FILES})