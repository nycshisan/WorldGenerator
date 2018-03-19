//
// Created by Nycshisan on 2018/3/15.
//

#ifndef WORLDGENERATOR_VORONOI_H
#define WORLDGENERATOR_VORONOI_H

#include "delaunay.h"

#include <vector>

class VoronoiDiagram {
public:
    struct VertexNode {
        std::vector<int> edgeIds;
        Point pos;

        VertexNode() = default;
        explicit VertexNode(const Point &p);
    };

    struct EdgeNode {
        Point endPoints[2];
        int relatedCenterIds[2] = {-1, -1};
        sf::Vertex vertex[2];

        EdgeNode() = default;
        EdgeNode(const Point &pa, const Point &pb);
    };

    typedef DelaunayTriangles::Output InputTris;
    typedef BlockCenters::Output InputCenters;
    typedef std::pair<std::map<int, VertexNode>, std::map<int, EdgeNode>> Output;
private:
    InputCenters _centers;
    InputTris _tris;
    Output _diagram;

    int _newEdgeId = 0;
    sf::CircleShape _pointShape;

    unsigned int _width, _height;

    Line _boxEdge[4];

public:
    void input(InputCenters centers, InputTris tris);
    void generate();
    Output output();
    void draw(Window &window);
};

#endif //WORLDGENERATOR_VORONOI_H
