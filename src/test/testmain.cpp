//
// Created by Nycshisan on 2018/4/8.
//

#include "../graphics/window.h"

using namespace wg;

void testVD() {
    auto window = MainWindow::MakeWindow();

    int testNumber = 200;
    bool loadData = false;

    for (int i = 1; i <= testNumber; ++i) {
        if (i % (testNumber / 100) == 0) {
            LOG("Proceeding: " + std::to_string(i) + "/" + std::to_string(testNumber));
        }
        Generator::NextButtonResponder(*window); // Block centers
        if (loadData) {
            Generator::LoadButtonResponder(*window);
        }
        Generator::NextButtonResponder(*window); // Delaunay triangles
        Generator::NextButtonResponder(*window); // Voronoi diagram
        Generator::NextButtonResponder(*window); // Lloyd relaxation
        Generator::UndoButtonResponder(*window);
        Generator::UndoButtonResponder(*window);
        Generator::UndoButtonResponder(*window);
        Generator::UndoButtonResponder(*window);
    }

    LOG("OK!");

}

void testStates(int target) {
    auto window = MainWindow::MakeWindow();
    while (Generator::SharedInstance().state < target)
        Generator::NextButtonResponder(*window);
}

int main() {
    testStates(Generator::State::LloydRelaxation);
    return 0;
}