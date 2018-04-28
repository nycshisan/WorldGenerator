//
// Created by Nycshisan on 2018/4/4.
//

#ifndef WORLDGENERATOR_LLOYD_H
#define WORLDGENERATOR_LLOYD_H

#include "voronoi.h"

class LloydRelaxation {
public:
    typedef VoronoiDiagram::Output Input;
    typedef std::pair<std::map<int, VoronoiDiagram::CenterNode>, std::map<int, VoronoiDiagram::EdgeNode>> Output;

private:
    Input _inputVd;
    Output _relaxedVd;

public:
    void input(Input vd);
    void generate();
    Output output();
    void draw(Drawer &drawer);
};

#endif //WORLDGENERATOR_LLOYD_H
