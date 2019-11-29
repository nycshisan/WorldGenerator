cmake_minimum_required(VERSION 3.8)
project(WorldGenerator)

set(CMAKE_CXX_STANDARD 11)

include_directories(lib/rapidjson1.1.0)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    include_directories(lib/win32/SFML-2.5.1/include)
    link_directories(lib/win32/SFML-2.5.1/lib)
    link_libraries(libsfml-audio.a libsfml-graphics.a libsfml-main.a libsfml-network.a libsfml-system.a libsfml-window.a)
elseif (APPLE)
    include_directories(lib/apple/sfml2.4.2/include)
    link_directories(lib/apple/sfml2.4.2/lib)
    link_libraries(libsfml-window.dylib libsfml-system.dylib libsfml-audio.dylib libsfml-graphics.dylib libsfml-network.dylib)
endif()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(SOURCE_FILES src/conf/conf.cpp src/conf/conf.h src/graphics/window.cpp src/graphics/window.h src/misc/log.h src/graphics/button.cpp src/graphics/button.h src/generator/generator.cpp src/generator/generator.h src/generator/impl/voronoi.cpp src/generator/impl/voronoi.h src/data/geomath.cpp src/data/geomath.h src/generator/impl/centers.cpp src/generator/impl/centers.h src/generator/impl/delaunay.cpp src/generator/impl/delaunay.h src/misc/simd.h src/misc/misc.h src/generator/impl/lloyd.cpp src/generator/impl/lloyd.h src/misc/error.cpp src/misc/error.h src/generator/impl/coast.cpp src/generator/impl/coast.h src/generator/impl/blocks.cpp src/generator/impl/blocks.h src/data/blockInfo.h src/graphics/drawer.cpp src/graphics/drawer.h src/data/blockinfo.cpp src/misc/noise.cpp src/misc/noise.h src/graphics/gfxmath.cpp src/graphics/gfxmath.h src/graphics/configWindow.cpp src/graphics/configWindow.h src/generator/config.h src/generator/config.cpp src/generator/impl.cpp src/generator/impl.h)
add_executable(WorldGenerator src/main.cpp ${SOURCE_FILES})

set(TEST_SOURCE_FILES src/test/testmain.cpp)
add_executable(Test ${SOURCE_FILES} ${TEST_SOURCE_FILES})