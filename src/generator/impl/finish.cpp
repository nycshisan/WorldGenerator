//
// Created by nycsh on 2020/2/2.
//

#include "finish.h"

std::string wg::Finish::getHintLabelText() {
    return "Finished!";
}

void wg::Finish::input(void *inputData) {}

void wg::Finish::generate() {}

void *wg::Finish::output() {
    return nullptr;
}

void wg::Finish::prepareVertexes(wg::Drawer &drawer) {}
