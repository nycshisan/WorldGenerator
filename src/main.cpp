#include <iostream>

#include "conf/conf.h"
#include "graphics/window.h"

int main() {
    int width = CONF.getMapWidth();
    int height = CONF.getMapHeight();
    int barHeight = CONF.getUIBarHeight();

    Window window(width, height, barHeight);

    window.play();

    return 0;
}