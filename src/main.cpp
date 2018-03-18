#include <iostream>

#include "conf/conf.h"
#include "graphics/window.h"

int main() {
    const Configure &conf = CONF;
    unsigned int width = conf["map"]["width"].GetUint();
    unsigned int height = conf["map"]["height"].GetUint();
    unsigned int barHeight = conf["ui"]["barHeight"].GetUint();

    Window window(width, height, barHeight);

    window.play();
    return 0;
}