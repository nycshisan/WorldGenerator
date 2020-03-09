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

    int size = CONF.getDistFieldSize();
    auto width = size, height = size;

    _t.create(width, height);
    _s.setTexture(_t);
    float scaleX = float(CONF.getUIMapWidth()) / float(width), scaleY = float(CONF.getUIMapHeight()) / float(width);
    _s.setScale(scaleX, scaleY);

    _dfx = new float[width * height];
    _dfy = new float[width * height];

    auto handle = CMJFAHandleAlloc(size);
    CMJFAInitPoint points[2];
    points[0].x = points[0].y = size / 3 * 2;
    points[0].vx = points[0].vy = 0.f;
    points[1].x = points[1].y = size / 3;
    points[1].vx = points[1].vy = 0.f;
    CMJFAInit(handle, points, 2);

    CMJFACalculate(handle, _dfx, _dfy);

    auto rgba = new unsigned char[width * height * 4];

    CMJFAGenerateTexture(handle, rgba);

    _t.update(rgba);
    delete[] rgba;

    CMJFAHandleFree(handle);

    _outputData = (void*)&_blockInfos;
}

void wg::DistField::prepareVertexes(wg::Drawer &drawer) {
    drawer.addSprite(_s);
}
