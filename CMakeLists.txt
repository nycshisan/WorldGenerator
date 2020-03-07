cmake_minimum_required(VERSION 3.8)
project(WorldGenerator)

set(CMAKE_CXX_STANDARD 17)

include_directories(lib/rapidjson1.1.0)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    include_directories(lib/win32/SFML-2.5.1/include)
    link_directories(lib/win32/SFML-2.5.1/lib)
    link_libraries(libsfml-audio.a libsfml-graphics.a libsfml-main.a libsfml-network.a libsfml-system.a libsfml-window.a)

    include_directories(src/cuda_modules/src)
    link_directories(src/cuda_modules/x64/Release)
    link_libraries(cuda_modules.lib)
elseif (APPLE)
    include_directories(lib/apple/sfml2.4.2/include)
    link_directories(lib/apple/sfml2.4.2/lib)
    link_libraries(libsfml-window.dylib libsfml-system.dylib libsfml-audio.dylib libsfml-graphics.dylib libsfml-network.dylib)
endif()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

file(GLOB SOURCE_FILES
        src/conf/*.cpp
        src/data/*.cpp
        src/generator/*.cpp
        src/generator/impl/*.cpp
        src/graphics/*.cpp
        src/misc/*.cpp
        )

add_executable(WorldGenerator src/main.cpp ${SOURCE_FILES})

add_executable(Test src/test/testmain.cpp ${SOURCE_FILES})