//
// Created by nycsh on 2020/3/7.
//

#include "distField.h"

#include "cuda_modules/jfa/jfa.h"

#include "blockEdges.h"

std::string wg::DistField::getHintLabelText() {
    return "Block edges distance field generated.";
}

void wg::DistField::generate() {
    delete[] _dfx;
    delete[] _dfy;

    _blockInfos = *(BlockEdges::Output*)_inputData;

    unsigned width = 2048, height = 2048;

    _t.create(width, height);
    _s.setTexture(_t);
    float scaleX = float(CONF.getUIMapWidth()) / float(width), scaleY = float(CONF.getUIMapHeight()) / float(width);
    _s.setScale(scaleX, scaleY);

    _dfx = new float[width * height];
    _dfy = new float[width * height];
    float large = float(width) * float(height);
    large *= large;
    // initialize distance field
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            _dfx[i * width + j] = large;
            _dfy[i * width + j] = large;
        }
    }
    _dfx[width * (height / 2) + width / 2] = _dfy[width * (height / 2) + width / 2] = 0.f;

    CMJFACalculate(_dfx, _dfy, int(width));
    auto stat = CMJFAGetStat();

    auto rgba = new unsigned char[width * height * 4];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int base = int(i * width + j);
            float x = _dfx[base], y = _dfy[base];
            float relDist = std::hypotf(x, y) / stat.maxDist;
            unsigned char color = std::lroundf(relDist * 255);

            base *= 4;

            rgba[base + 0] = color;
            rgba[base + 1] = color;
            rgba[base + 2] = color;
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
