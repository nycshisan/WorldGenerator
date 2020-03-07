//
// Created by nycsh on 2020/2/2.
//

#include "finish.h"

std::string wg::Finish::getHintLabelText() {
    return "Finished!";
}

void wg::Finish::generate() {
    if (CONF.getInstallEnable()) {
        // install blocks data
        _installFile("blocks");
        // install block edges distance field
        _installFile("distField");
    }
}

void wg::Finish::prepareVertexes(wg::Drawer &drawer) {}

bool wg::Finish::_installFile(const std::string &moduleName) {
    return CopyFile(CONF.getOutputDirectory() + CONF.getModuleOutputPath(moduleName),
                    CONF.getInstallTarget() + CONF.getModuleOutputPath(moduleName));
}
