#include <iostream>

#include "conf/conf.h"
#include "graphics/window.h"
#include "misc/log.h"

int main() {
    const Configure &conf = Configure::SharedInstance();
    unsigned int width = conf["map"]["width"].GetUint();
    unsigned int height = conf["map"]["height"].GetUint();
    unsigned int barHeight = conf["ui"]["barHeight"].GetUint();

    Window window(width, height, barHeight);

    window.play();

    return 0;
}