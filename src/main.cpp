#include "conf/conf.h"
#include "graphics/window.h"
#include "generator/generator.h"

int main() {
    int width = CONF.getMapWidth();
    int height = CONF.getMapHeight();
    int barHeight = CONF.getUIBarHeight();

    wg::Window window(width, height, barHeight);

//    wg::Generator::NextButtonResponder(window);
//    wg::Generator::NextButtonResponder(window);
//    wg::Generator::NextButtonResponder(window);
//    wg::Generator::NextButtonResponder(window);
//    wg::Generator::NextButtonResponder(window);

    window.play();

    return 0;
}