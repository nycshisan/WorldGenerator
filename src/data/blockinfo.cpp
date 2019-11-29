//
// Created by Nycshisan on 2018/4/25.
//

#include "blockInfo.h"

#include <fstream>

namespace wg {

    void BlockInfo::addMarginEdge(const Rectangle &box) {
        std::vector<std::pair<std::shared_ptr<VertexInfo>, Line>> edgePoints;
        for (auto &edgeInfo: edges) {
            for (auto &vertexInfo: edgeInfo->vertexes) {
                Point &p = vertexInfo->point;
                Line l;
                if (box.onEdge(p, l)) {
                    edgePoints.emplace_back(std::pair<std::shared_ptr<VertexInfo>, Line>(vertexInfo, l));
                };
            }
        }
        if (!edgePoints.empty()) {
            auto &va = edgePoints[0].first, &vb = edgePoints[1].first;
            auto &la = edgePoints[0].second, &lb = edgePoints[1].second;
            if ((la.vertical && lb.vertical) || (la.horizontal && lb.horizontal)) {
                auto edge = std::make_shared<EdgeInfo>(-1);
                edge->isMargin = true;
                edge->relatedBlocks.emplace(thisPtr);
                edge->vertexes.emplace(va);
                edge->vertexes.emplace(vb);
                edges.emplace(edge);
            } else {
                if (la.horizontal) {
                    std::swap(la, lb);
                }
                auto corner = std::make_shared<VertexInfo>(-1);
                corner->isCorner = true;
                corner->point = Point(la.verticalX, lb.horizontalY);
                vertexes.emplace(corner);

                auto edgeA = std::make_shared<EdgeInfo>(-1);
                edgeA->isMargin = true;
                edgeA->relatedBlocks.emplace(thisPtr);
                edgeA->vertexes.emplace(va);
                edgeA->vertexes.emplace(corner);
                edges.emplace(edgeA);

                auto edgeB = std::make_shared<EdgeInfo>(-1);
                edgeB->isMargin = true;
                edgeB->relatedBlocks.emplace(thisPtr);
                edgeB->vertexes.emplace(vb);
                edgeB->vertexes.emplace(corner);
                edges.emplace(edgeB);

                corner->relatedEdges.emplace(edgeA);
                corner->relatedEdges.emplace(edgeB);
                corner->relatedBlocks.emplace(thisPtr);
            }
        }
    }

    void BlockInfo::calcArea() {
        for (auto &edge: edges) {
            assert(edge->vertexes.size() == 2);
            const Point &p1 = (*edge->vertexes.begin())->point, &p2 = (*edge->vertexes.rbegin())->point;
            area += std::abs((p1.x - center.x) * (p2.y - center.y) - (p2.x - center.x) * (p1.y - center.y));
        }
        area /= 2;
    }

    void BlockInfo::SaveBlockInfosTo(std::ofstream &ofs, const std::vector<std::shared_ptr<BlockInfo>> &infos) {
        int head = 'WGBI';
        ofs.write((char*)&head, sizeof(int));
        size_t infosSize = infos.size();
        ofs.write((char*)&infosSize, sizeof(size_t));
        for (size_t i = 0; i < infosSize; ++i) {

        }
    }

}