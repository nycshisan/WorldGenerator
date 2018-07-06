#include "conf/conf.h"
#include "graphics/window.h"
#include "generator/generator.h"
#include "misc/misc.h"

int main() {
    int width = CONF.getMapWidth();
    int height = CONF.getMapHeight();

    wg::MainWindow window(width, height);

    window.play();

    return 0;
}