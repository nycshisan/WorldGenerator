//
// Created by Nycshisan on 2018/3/15.
//

#ifndef WORLDGENERATOR_VORONOI_H
#define WORLDGENERATOR_VORONOI_H

#include <vector>

#include "centers.h"
#include "delaunay.h"

namespace wg {

    class VoronoiDiagram {
    public:
        struct CenterNode {
            std::vector<int> edgeIds;
            Point point;

            CenterNode() = default;

            explicit CenterNode(const Point &p);
        };

        struct EdgeNode {
            int relatedCenterIds[2] = {-1, -1};
            Point vertex[2];
            int vertexIds[2] = {-1, -1};

            EdgeNode() = default;

            EdgeNode(const Point &pa, int paId, const Point &pb, int pbId);
        };

        typedef DelaunayTriangles::Output InputTris;
        typedef Centers::Output InputCenters;
        typedef std::pair<std::map<int, CenterNode>, std::map<int, EdgeNode>> Output;
    private:
        InputCenters _centers;
        InputTris _tris;
        Output _diagram;

        std::map<int, std::map<int, bool>> _existsEdges;

        bool _existsEdge(int paId, int pbId);

    public:
        void input(const InputCenters &centers, const InputTris &tris);

        void generate();

        Output output();

        void prepareVertexes(Drawer &drawer);
    };

}

#endif //WORLDGENERATOR_VORONOI_H
