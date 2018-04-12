cmake_minimum_required(VERSION 3.8)
project(WorldGenerator)

set(CMAKE_CXX_STANDARD 11)

include_directories(lib/rapidjson1.1.0 lib/sfml2.4.2/include)

link_directories(lib/sfml2.4.2/lib)
link_libraries(libsfml-window.dylib libsfml-system.dylib libsfml-audio.dylib libsfml-graphics.dylib libsfml-network.dylib)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(SOURCE_FILES src/conf/conf.cpp src/conf/conf.h src/graphics/window.cpp src/graphics/window.h src/misc/log.h src/graphics/button.cpp src/graphics/button.h src/generator/generator.cpp src/generator/generator.h src/generator/voronoi.cpp src/generator/voronoi.h src/misc/geomath.cpp src/misc/geomath.h src/generator/centers.cpp src/generator/centers.h src/generator/delaunay.cpp src/generator/delaunay.h src/misc/simd.cpp src/misc/simd.h src/misc/misc.h src/generator/lloyd.cpp src/generator/lloyd.h src/misc/error.cpp src/misc/error.h)
add_executable(WorldGenerator src/main.cpp ${SOURCE_FILES})

set(TEST_SOURCE_FILES src/test/testmain.cpp)
add_executable(Test ${SOURCE_FILES} ${TEST_SOURCE_FILES})