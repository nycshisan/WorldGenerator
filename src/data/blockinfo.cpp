//
// Created by Nycshisan on 2018/4/25.
//

#include "blockInfo.h"

#include <unordered_set>
#include <unordered_map>

#include "binaryIO.h"
#include "../conf/conf.h"

static unsigned int MarginEdgeId;
static unsigned int CornerVertexId;

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
                auto edge = std::make_shared<EdgeInfo>(MarginEdgeId++);
                edge->isMargin = true;
                edge->relatedBlocks.emplace(thisPtr);
                edge->vertexes.emplace(va);
                edge->vertexes.emplace(vb);
                edges.emplace(edge);
            } else {
                if (la.horizontal) {
                    std::swap(la, lb);
                }
                auto corner = std::make_shared<VertexInfo>(CornerVertexId++);
                corner->isCorner = true;
                corner->point = Point(la.verticalX, lb.horizontalY);
                vertexes.emplace(corner);

                auto edgeA = std::make_shared<EdgeInfo>(MarginEdgeId++);
                edgeA->isMargin = true;
                edgeA->relatedBlocks.emplace(thisPtr);
                edgeA->vertexes.emplace(va);
                edgeA->vertexes.emplace(corner);
                edges.emplace(edgeA);

                auto edgeB = std::make_shared<EdgeInfo>(MarginEdgeId++);
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

    static std::string BlockInfoVerifyHead = "WGBIHD"; // NOLINT(cert-err58-cpp)
    static std::string BlockInfoVerifyTail = "WGBITL"; // NOLINT(cert-err58-cpp)

    void BlockInfo::SaveBlockInfosTo(std::ofstream &ofs, const std::vector<std::shared_ptr<BlockInfo>> &infos) {
        using namespace BinaryIO;

        std::unordered_set<std::shared_ptr<EdgeInfo>> edges;
        for (const auto &info : infos) {
            for (const auto &edgePtr : info->edges) {
                edges.insert(edgePtr);
            }
        }
        std::unordered_set<std::shared_ptr<VertexInfo>> vertexes;
        for (const auto &info : infos) {
            for (const auto &vertexPtr : info->vertexes) {
                vertexes.insert(vertexPtr.lock());
            }
        }

        write(ofs, BlockInfoVerifyHead);
        write(ofs, CONF.getMapWidth());
        write(ofs, CONF.getMapHeight());
        write(ofs, infos.size());
        write(ofs, edges.size());
        write(ofs, vertexes.size());
        for (const auto &ele : infos) {
            write(ofs, ele);
        }
        for (const auto &ele : edges) {
            write(ofs, ele);
        }
        for (const auto &ele : vertexes) {
            write(ofs, ele);
        }
        write(ofs, BlockInfoVerifyTail);
    }

    void BlockInfo::LoadBlockInfosTo(std::ifstream &ifs, std::vector<std::shared_ptr<BlockInfo>> &infos) {
        using namespace BinaryIO;

        std::string head;
        read(ifs, head, BlockInfoVerifyHead.size());
        assert(head == BlockInfoVerifyHead);

        size_t blockNum, edgeNum, vertexNum;
        int width, height;
        read(ifs, width); read(ifs, height);
        read(ifs, blockNum);
        read(ifs, edgeNum);
        read(ifs, vertexNum);

        std::vector<std::shared_ptr<EdgeInfo>> edges;
        std::vector<std::shared_ptr<VertexInfo>> vertexes;

        read(ifs, infos, blockNum);
        read(ifs, edges, edgeNum);
        read(ifs, vertexes, vertexNum);

        std::unordered_map<decltype(BlockInfo::id), std::shared_ptr<BlockInfo>> bm;
        std::unordered_map<decltype(EdgeInfo::id), std::shared_ptr<EdgeInfo>> em;
        std::unordered_map<decltype(VertexInfo::id), std::shared_ptr<VertexInfo>> vm;
        for (const auto &ptr : infos) {
            bm[ptr->id] = ptr;
        }
        for (const auto &ptr : edges) {
            em[ptr->id] = ptr;
        }
        for (const auto &ptr : vertexes) {
            vm[ptr->id] = ptr;
        }

        for (const auto &ptr : vertexes) {
            for (int id : ptr->edgeIds) {
                ptr->relatedEdges.emplace(em[id]);
            }
            for (int id : ptr->blockIds) {
                ptr->relatedBlocks.emplace(bm[id]);
            }
        }
        for (const auto &ptr : edges) {
            for (int id : ptr->blockIds) {
                ptr->relatedBlocks.emplace(bm[id]);
            }
            for (int id : ptr->vertexIds) {
                ptr->vertexes.emplace(vm[id]);
            }
        }
        for (const auto &ptr : infos) {
            for (int id : ptr->edgeIds) {
                ptr->edges.emplace(em[id]);
            }
            for (int id : ptr->vertexIds) {
                ptr->vertexes.emplace(vm[id]);
            }
            ptr->thisPtr = ptr;
            ptr->calcArea();
        }

        std::string tail;
        read(ifs, tail, BlockInfoVerifyTail.size());
        assert(tail == BlockInfoVerifyTail);
    }

    void BlockInfo::ReinitHelperId() {
        MarginEdgeId = 1000000;
        CornerVertexId = 2000000;
    }

    sf::Vector2f EdgeInfo::sample(float t) {
        return (*vertexes.begin())->point * (1.f - t) + (*vertexes.rbegin())->point * t;
    }
}