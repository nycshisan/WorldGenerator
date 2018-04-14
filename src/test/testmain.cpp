//
// Created by Nycshisan on 2018/4/8.
//

#include "../conf/conf.h"
#include "../graphics/window.h"
#include "../generator/generator.h"

int main() {
    int width = CONF.getMapWidth();
    int height = CONF.getMapHeight();
    int barHeight = CONF.getUIBarHeight();

    Window window(width, height, barHeight);

    int testNumber = 200;
    bool loadData = false;

    for (int i = 1; i <= testNumber; ++i) {
        if (i % (testNumber / 100) == 0) {
            LOGOUT("Proceeding: " + std::to_string(i) + "/" + std::to_string(testNumber));
        }
        Generator::NextButtonResponder(window); // Block centers
        if (loadData) {
            Generator::LoadButtonResponder(window);
        }
        Generator::NextButtonResponder(window); // Delaunay triangles
        Generator::NextButtonResponder(window); // Voronoi diagram
        Generator::NextButtonResponder(window); // Lloyd relaxation
        Generator::UndoButtonResponder(window);
        Generator::UndoButtonResponder(window);
        Generator::UndoButtonResponder(window);
        Generator::UndoButtonResponder(window);
    }

    LOGOUT("OK!");

    return 0;
}