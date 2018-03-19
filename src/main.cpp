#include <iostream>

#include "conf/conf.h"
#include "graphics/window.h"

int main() {
    unsigned int width = CONF.getMapWidth();
    unsigned int height = CONF.getMapHeight();
    unsigned int barHeight = CONF.getUIBarHeight();

    Window window(width, height, barHeight);

    window.play();

    return 0;
}