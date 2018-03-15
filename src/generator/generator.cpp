//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

void NextButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready:
            window.setHintLabel("Generate ");
            window.setHintLabelDone();
            generator._state = PointsPined;
            break;
        default:
            LOGERR("Invalid generator state!");
    }
}

void RedoButtonResponder(Window &window) {

}

void SaveButtonResponder(Window &window) {}

Generator &Generator::SharedInstance() {
    static Generator instance;
    return instance;
}
