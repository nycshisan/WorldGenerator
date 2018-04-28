//
// Created by Nycshisan on 2018/4/15.
//

#include "blocks.h"

#include "../conf/conf.h"

void Blocks::input(Blocks::Input relaxedVd) {
    _relaxedVd = std::move(relaxedVd);
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
        blockInfo->center = center.pos;

        for (auto edgeId: center.edgeIds) {
            if (initializedEdgeInfos.count(edgeId) == 0) {
                // Initialize the edge information
                auto &edge = edgeMap[edgeId];
                auto newEdgeInfo = std::make_shared<EdgeInfo>(edgeId);
                for (int i = 0 ; i < 2; ++i) {
                    int vertexId = edge.vertexIds[i];
                    if (initializedVertexInfos.count(vertexId) == 0) {
                        // Initialize the vertex information
                        auto &vertex = edge.vertex[i];
                        auto newVertexInfo = std::make_shared<VertexInfo>(vertexId);
                        newVertexInfo->pos = vertex;
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

void Blocks::draw(Drawer &drawer) {
    for (auto &blockInfo: _blockInfos) {
        drawer.draw(*blockInfo);
    }
}
