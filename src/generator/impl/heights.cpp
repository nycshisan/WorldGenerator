//
// Created by nycsh on 2020/2/10.
//

#include "heights.h"

#include "blockEdges.h"

namespace wg {

    void Heights::generate() {
        auto &blockInfos = *(BlockEdges::Output*)_inputData;

        auto heightMapWidth = CONF.getHeightMapWidth(), heightMapHeight = CONF.getHeightMapHeight();

        _blockHeightInfos.first = blockInfos;
        _blockHeightInfos.second.resize(heightMapHeight, std::vector<float>(heightMapWidth, 0.5f));

        BlockHeightInfoDrawable::setTexture(_blockHeightInfos.second);

        _outputData = (void*)&_blockHeightInfos;
    }

    void Heights::prepareVertexes(Drawer &drawer) {
        BlockHeightInfoDrawable::drawHeightsTo(drawer, _blockHeightInfos.first);
    }

    std::string Heights::getHintLabelText() {
        return "Initialized the height map.";
    }
}