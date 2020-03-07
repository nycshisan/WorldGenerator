//
// Created by nycsh on 2020/3/7.
//

#include "distField.h"

#include "blockEdges.h"

std::string wg::DistField::getHintLabelText() {
    return "Block edges distance field generated.";
}

void wg::DistField::generate() {
    _blockInfos = *(BlockEdges::Output*)_inputData;

    _outputData = (void*)&_blockInfos;
}

void wg::DistField::prepareVertexes(wg::Drawer &drawer) {

}
