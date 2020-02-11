//
// Created by nycsh on 2020/2/10.
//

#include "heights.h"

#include "../generator.h"

namespace wg {

    void Heights::input(void* inputData) {
        _blockInfos = *(Input*)inputData;
    }

    void Heights::generate() {
        _blockHeightInfos.first = _blockInfos;
    }

    void *Heights::output() {
        return (void*)&_blockHeightInfos;
    }

    void Heights::prepareVertexes(Drawer &drawer) {
        
    }

    std::string Heights::getHintLabelText() {
        return "Initialize the height map.";
    }
}