//
// Created by Nycshisan on 2018/4/15.
//

#include "blocks.h"

#include <cmath>

#include "../conf/conf.h"
#include "../graphics/drawer.h"

namespace wg {

    void Blocks::input(const Input &relaxedVd) {
        _relaxedVd = relaxedVd;
    }

    void Blocks::generate() {
        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        Rectangle box = Rectangle(0, width, height, 0);

        _blockInfos.clear();

        auto &centerMap = _relaxedVd.first;
        auto &edgeMap = _relaxedVd.second;

        std::map<int, std::shared_ptr<EdgeInfo>> initializedEdgeInfos;
        std::map<int, std::shared_ptr<VertexInfo>> initializedVertexInfos;

        for (auto &pair: centerMap) {
            // Initialize the block information
            auto &center = pair.second;
            auto blockInfo = std::make_shared<BlockInfo>(pair.first);
            blockInfo->thisPtr = blockInfo;
            blockInfo->center = center.point;

            for (auto edgeId: center.edgeIds) {
                if (initializedEdgeInfos.count(edgeId) == 0) {
                    // Initialize the edge information
                    auto &edge = edgeMap[edgeId];
                    auto newEdgeInfo = std::make_shared<EdgeInfo>(edgeId);
                    for (int i = 0; i < 2; ++i) {
                        int vertexId = edge.vertexIds[i];
                        if (initializedVertexInfos.count(vertexId) == 0) {
                            // Initialize the vertex information
                            auto &vertex = edge.vertex[i];
                            auto newVertexInfo = std::make_shared<VertexInfo>(vertexId);
                            newVertexInfo->point = vertex;
                            initializedVertexInfos[vertexId] = newVertexInfo;
                        }
                        auto vertexInfo = initializedVertexInfos[vertexId];
                        newEdgeInfo->vertexes.emplace(vertexInfo);
                        vertexInfo->relatedEdges.emplace(newEdgeInfo);
                    }
                    initializedEdgeInfos[edgeId] = newEdgeInfo;
                }
                auto &edgeInfo = initializedEdgeInfos[edgeId];
                for (auto &vertex: edgeInfo->vertexes) {
                    blockInfo->vertexes.emplace(vertex);
                    vertex->relatedBlocks.emplace(blockInfo);
                }
                blockInfo->edges.emplace(edgeInfo);
                edgeInfo->relatedBlocks.emplace(blockInfo);
            }
            blockInfo->addMarginEdge(box);
            blockInfo->calcArea();
            _blockInfos.emplace_back(blockInfo);
        }

        assertWithSave(_blockInfos.size() == CONF.getCenterNumber());
    }

    Blocks::Output Blocks::output() {
        return _blockInfos;
    }

    void Blocks::prepareVertexes(Drawer &drawer) {
        for (auto &blockInfo: _blockInfos) {
            sf::Uint8 r = 0, g = 0, b = 0;
            r += sf::Uint8(blockInfo->id);
            for (auto &edgeInfo: blockInfo->edges) {
                g += edgeInfo->id;
            }
            for (auto &vertexInfo: blockInfo->vertexes) {
                b += vertexInfo.lock()->id;
            }
            auto color = sf::Color(r, g, b);

            _prepareBlockVertexes(drawer, blockInfo, color);
        }
    }

    void BlocksDrawable::_prepareBlockVertexes(Drawer &drawer, const std::shared_ptr<BlockInfo> &blockInfo,
                                               const sf::Color &color) {
        sf::Vertex v0 = sf::Vertex(blockInfo->center, color);
        for (auto &edgeInfo: blockInfo->edges) {
            drawer.appendVertex(sf::Triangles, v0);
            for (auto &vertexInfo: edgeInfo->vertexes) {
                drawer.appendVertex(sf::Triangles, sf::Vertex(vertexInfo->point, color));
            }
        }
    }

    void BlocksDrawable::_prepareCoast(Drawer &drawer, const std::shared_ptr<EdgeInfo> &edgeInfo) {
        Point pointA = (*edgeInfo->vertexes.begin())->point, pointB = (*edgeInfo->vertexes.rbegin())->point;
        float length = std::sqrt(pointA.squareDistance(pointB));
        float sin = (pointA.y - pointB.y) / length, cos = (pointA.x - pointB.x) / length;
        Point ltp(pointB.x - _CoastThickness * sin, pointB.y + _CoastThickness * cos),
                rtp(pointB.x + _CoastThickness * sin, pointB.y - _CoastThickness * cos),
                lbp(pointA.x - _CoastThickness * sin, pointA.y + _CoastThickness * cos),
                rbp(pointA.x + _CoastThickness * sin, pointA.y - _CoastThickness * cos);
        drawer.appendVertex(sf::Triangles, ltp);
        drawer.appendVertex(sf::Triangles, lbp);
        drawer.appendVertex(sf::Triangles, rtp);
        drawer.appendVertex(sf::Triangles, rtp);
        drawer.appendVertex(sf::Triangles, lbp);
        drawer.appendVertex(sf::Triangles, rbp);
    }
}
