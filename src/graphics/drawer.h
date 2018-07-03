//
// Created by Nycshisan on 2018/4/16.
//

#ifndef WORLDGENERATOR_DRAWER_H
#define WORLDGENERATOR_DRAWER_H

#include "window.h"
#include "../data/blockInfo.h"

class Drawer {
    Window *_window;

    sf::CircleShape _pointShape;

    enum DrawBufferType {
        PointBuf,
        LineBuf,
        TriangleBuf
    };

    int _drawBufSize = 128;
    DrawBufferType _crtBufType;

public:
    explicit Drawer(Window *window);

    void draw(const Point &point);
    void draw(const Point &point1, const Point &point2);
    void draw(const Triangle &tri);
    void draw(const BlockInfo &blockInfo);
    void draw(const BlockInfo &blockInfo, const sf::Color &color);
    void draw(const EdgeInfo &edgeInfo);

    void commit();
};

#endif //WORLDGENERATOR_DRAWER_H
