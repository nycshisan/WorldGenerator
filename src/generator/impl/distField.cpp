//
// Created by nycsh on 2020/3/7.
//

#include "distField.h"

#include "blockEdges.h"

std::string wg::DistField::getHintLabelText() {
    return "Block edges distance field generated.";
}

void wg::DistField::generate() {
    delete[] _df;

    _blockInfos = *(BlockEdges::Output*)_inputData;

    unsigned width = 64, height = 64;

    _t.create(width, height);
    _s.setTexture(_t);
    float scaleX = float(CONF.getUIMapWidth()) / float(width), scaleY = float(CONF.getUIMapHeight()) / float(width);
    _s.setScale(scaleX, scaleY);

    _df = new float[width * height * 2];

    auto rgba = new unsigned char[width * height * 4];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int base = int(i * width + j) * 4;
            int base_df = base / 2;
            rgba[base + 0] = std::lround(_df[base_df + 0]) % 256;
            rgba[base + 1] = std::lround(_df[base_df + 1]) % 256;
            rgba[base + 2] = 0;
            rgba[base + 3] = 255;
        }
    }

    _t.update(rgba);
    delete[] rgba;

    _outputData = (void*)&_blockInfos;
}

void wg::DistField::prepareVertexes(wg::Drawer &drawer) {
    drawer.addSprite(_s);
}
