//
// Created by Nycshisan on 2018/3/15.
//

#ifndef WORLDGENERATOR_VORONOI_H
#define WORLDGENERATOR_VORONOI_H

#include "delaunay.h"

class VoronoiDiagram {
public:
    class VertexNode {

    };

    class EdgeNode {

    };

    typedef DelaunayTriangles::Output InputTris;
    typedef BlockCenters::Output InputCenters;
    typedef std::pair<std::map<int, VertexNode>, std::map<int, EdgeNode>> Output;
private:
    InputCenters _centers;
    InputTris _tris;
    Output _diagram;

public:
    void init(InputCenters centers, InputTris tris);
    void generate();
    Output output();
    void draw(Window &window);
};

#endif //WORLDGENERATOR_VORONOI_H
