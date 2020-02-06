//
// Created by nycsh on 2020/2/2.
//

#include "finish.h"

std::string wg::Finish::getHintLabelText() {
    return "Finished!";
}

void wg::Finish::input(void *inputData) {}

void wg::Finish::generate() {
    if (CONF.getInstallEnable()) {
        // install blocks data
        _installFile(CONF.getBlocksOutputPath());
        // install coast info
        _installFile(CONF.getCoastOutputPath());
    }
}

void *wg::Finish::output() {
    return nullptr;
}

void wg::Finish::prepareVertexes(wg::Drawer &drawer) {}

bool wg::Finish::_installFile(const std::string &fp) {
    return CopyFile(CONF.getOutputDirectory() + fp, CONF.getInstallTarget() + fp);
}
