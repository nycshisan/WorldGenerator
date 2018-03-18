//
// Created by Nycshisan on 2018/3/15.
//

#ifndef WORLDGENERATOR_VORONOI_H
#define WORLDGENERATOR_VORONOI_H

#include <vector>

#include "SFML/Graphics.hpp"

#include "../graphics/window.h"
#include "../misc/geomath.h"

struct VDDelaunayTriangleNetNode;

class VoronoiDiagram {
    unsigned int _width = 0, _height = 0;

    std::vector<Point> _centers;

    sf::CircleShape _pointShape;

    VDDelaunayTriangleNetNode* triNetHead;
    std::vector<VDDelaunayTriangleNetNode*> _allocatedNodes;

public:
    void init(unsigned int width, unsigned int height);

    void generateCenters();
    void generateDelaunayTriangles();

    void drawPointsToWindow(Window &window);
    void drawTrianglesToWindow(Window &window);
};

#endif //WORLDGENERATOR_VORONOI_H
