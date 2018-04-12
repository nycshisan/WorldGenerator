//
// Created by Nycshisan on 2018/4/4.
//

#ifndef WORLDGENERATOR_LLOYD_H
#define WORLDGENERATOR_LLOYD_H

#include "voronoi.h"

class LloydRelaxation {
public:
    typedef VoronoiDiagram::Output Input;
    typedef std::pair<std::map<int, VoronoiDiagram::VertexNode>, std::map<int, VoronoiDiagram::EdgeNode>> Output;

private:
    Input _inputVd;
    Output _relaxedVd;

    float _factor;
    int _iteration;

    Rectangle _box;

    sf::CircleShape _pointShape;

public:
    void input(Input vd);
    void generate();
    Output output();
    void draw(Window &window);
};

#endif //WORLDGENERATOR_LLOYD_H
