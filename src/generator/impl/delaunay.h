//
// Created by Nycshisan on 2018/3/19.
//

#ifndef WORLDGENERATOR_DELAUNAY_H
#define WORLDGENERATOR_DELAUNAY_H

#include "../impl.h"

namespace wg {

    class DelaunayTriangles : public GeneratorImpl {
    public:
        class NetNode;

        struct Edge {
            int pid[2] = {-1, -1};
            NetNode *nextTri = nullptr;
            int nextTriEdgeId = -1;

            Edge() = default;

            Edge(int pointIdA, int pointIdB);
        };

        class NetNode : public Triangle {
            friend class DelaunayTriangles;

            bool _visited = false;

            void _clearVisitFlag();

            void
            _findInfluenced(const Point &point, int beginEdgeId, std::unordered_set<NetNode *> &tris, std::vector<Edge *> &edges);

            void findInfluenced(const Point &point, std::unordered_set<NetNode *> &tris, std::vector<Edge *> &edges);

            bool _isBoundingTriangle;

            void _linkAnoTri(NetNode *anoTri, int edgeId, int anoEdgeId);

            float _minX, _minY, _maxX, _maxY;

            NetNode(int id, int pointIdA, int pointIdB, int pointIdC, const std::vector<Point> &centers, int n);

        public:
            int id;
            Edge edges[3];

            Point exCenter;
            float exRadius = 0;
        };

        typedef std::pair<std::vector<Point>, std::unordered_set<NetNode*>> Output;
    public:
        Output _centersTris;

        NetNode *_triNetHead = nullptr;

        void _deleteOldNodes();

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;

        ~DelaunayTriangles();
    };

}

#endif //WORLDGENERATOR_DELAUNAY_H
