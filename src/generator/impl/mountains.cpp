//
// Created by Nycshisan on 2020/2/12.
//

#include "mountains.h"

#include "heights.h"

std::string wg::Mountains::getHintLabelText() {
    return "Mountains generated.";
}

void wg::Mountains::generate() {
    _blockHeightInfos = *(Heights::Output*)_inputData;

    BlockHeightInfoDrawable::setTexture(_blockHeightInfos.second);

    _outputData = (void*)&_blockHeightInfos;
}

void wg::Mountains::prepareVertexes(wg::Drawer &drawer) {
    BlockHeightInfoDrawable::drawHeightsTo(drawer, _blockHeightInfos.first);
}
