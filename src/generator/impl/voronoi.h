//
// Created by Nycshisan on 2018/3/15.
//

#ifndef WORLDGENERATOR_VORONOI_H
#define WORLDGENERATOR_VORONOI_H

#include "../impl.h"

namespace wg {

    class VoronoiDiagram : public GeneratorImpl {
    public:
        struct CenterNode {
            std::vector<int> edgeIds;
            Point point;

            CenterNode() = default;

            explicit CenterNode(const Point &p);
        };

        struct EdgeNode {
            int relatedCenterIds[2] = {-1, -1};
            Point point[2];
            int vertexIds[2] = {-1, -1};

            EdgeNode() = default;

            EdgeNode(const Point &pa, int paId, const Point &pb, int pbId);
        };

        typedef std::pair<std::map<int, CenterNode>, std::map<int, EdgeNode>> Output;
    private:
        Output _diagram;

        std::map<int, std::map<int, bool>> _existsEdges;

        bool _existsEdge(int paId, int pbId);

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_VORONOI_H
