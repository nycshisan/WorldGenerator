//
// Created by Nycshisan on 2018/4/16.
//

#ifndef WORLDGENERATOR_BLOCKINFO_H
#define WORLDGENERATOR_BLOCKINFO_H

#include <memory>
#include <unordered_set>
#include <set>

#include "geomath.h"

namespace wg {

    struct BlockInfo;
    struct EdgeInfo;

    struct VertexInfo {
        unsigned int id;

        wg::Point point;
        std::set<std::weak_ptr<BlockInfo>, std::owner_less<std::weak_ptr<BlockInfo>>> relatedBlocks;
        std::set<std::weak_ptr<EdgeInfo>, std::owner_less<std::weak_ptr<EdgeInfo>>> relatedEdges;

        bool isCorner = false;

        // caches in loading
        std::vector<unsigned int> blockIds;
        std::vector<unsigned int> edgeIds;

        explicit VertexInfo() : id(0) {}
        explicit VertexInfo(unsigned int id) : id(id) {}
    };

    struct EdgeInfo {
        unsigned int id;

        std::set<std::weak_ptr<BlockInfo>, std::owner_less<std::weak_ptr<BlockInfo>>> relatedBlocks;
        std::set<std::shared_ptr<VertexInfo>, std::owner_less<std::shared_ptr<VertexInfo>>> vertexes;

        // caches in loading
        std::vector<unsigned int> blockIds;
        std::vector<unsigned int> vertexIds;

        bool isMargin = false;

        explicit EdgeInfo() : id(0) {}
        explicit EdgeInfo(unsigned int id) : id(id) {}
    };

    struct BlockInfo {
        unsigned int id;
        std::weak_ptr<BlockInfo> thisPtr;

        wg::Point center;
        std::set<std::shared_ptr<EdgeInfo>, std::owner_less<std::shared_ptr<EdgeInfo>>> edges;
        std::set<std::weak_ptr<VertexInfo>, std::owner_less<std::weak_ptr<VertexInfo>>> vertexes;

        float area = 0.0f;

        // caches in loading
        std::vector<unsigned int> edgeIds;
        std::vector<unsigned int> vertexIds;

        explicit BlockInfo() : id(0) {}
        explicit BlockInfo(unsigned int id) : id(id) {}

        void addMarginEdge(const wg::Rectangle &box);

        void calcArea();

        static void SaveBlockInfosTo(std::ofstream &ofs, const std::vector<std::shared_ptr<BlockInfo>>& infos);

        static void LoadBlockInfosTo(std::ifstream &ifs, std::vector<std::shared_ptr<BlockInfo>>& infos);
    };

}

#endif //WORLDGENERATOR_BLOCKINFO_H
