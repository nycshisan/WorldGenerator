//
// Created by Nycshisan on 2018/4/15.
//

#include "blocks.h"

#include "lloyd.h"

namespace wg {

    void Blocks::generate() {
        auto &relaxedVd = *(LloydRelaxation::Output*)_inputData;

        BlockInfo::ReinitHelperId();

        int width = CONF.getMapWidth(), height = CONF.getMapHeight();
        Rectangle box = Rectangle(0, float(width), float(height), 0);

        _blockInfos.clear();

        auto &centerMap = relaxedVd.first;
        auto &edgeMap = relaxedVd.second;

        std::map<int, std::shared_ptr<EdgeInfo>> _initializedEdgeInfos;
        std::map<int, std::shared_ptr<VertexInfo>> _initializedVertexInfos;

        for (auto &pair: centerMap) {
            // Initialize the block information
            auto &center = pair.second;
            auto blockInfo = std::make_shared<BlockInfo>(pair.first);
            blockInfo->thisPtr = blockInfo;
            blockInfo->center = center.point;

            for (auto edgeId: center.edgeIds) {
                if (_initializedEdgeInfos.count(edgeId) == 0) {
                    // Initialize the edge information
                    auto &edge = edgeMap[edgeId];
                    auto newEdgeInfo = std::make_shared<EdgeInfo>(edgeId);
                    for (int i = 0; i < 2; ++i) {
                        int vertexId = edge.vertexIds[i];
                        if (_initializedVertexInfos.count(vertexId) == 0) {
                            // Initialize the vertex information
                            auto &vertex = edge.point[i];
                            auto newVertexInfo = std::make_shared<VertexInfo>(vertexId);
                            newVertexInfo->point = vertex;
                            _initializedVertexInfos[vertexId] = newVertexInfo;
                        }
                        auto vertexInfo = _initializedVertexInfos[vertexId];
                        newEdgeInfo->vertexes.emplace(vertexInfo);
                        vertexInfo->relatedEdges.emplace(newEdgeInfo);
                    }
                    _initializedEdgeInfos[edgeId] = newEdgeInfo;
                }
                auto &edgeInfo = _initializedEdgeInfos[edgeId];
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

        // set colors
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
//            std::cout << int(r) << " " << int(g) << " " << int(b) << std::endl;
            blockInfo->center.vertex.color = color;
        }

        _outputData = (void*)&_blockInfos;
    }

    void Blocks::prepareVertexes(Drawer &drawer) {
        for (auto &blockInfo: _blockInfos) {
            sf::Vertex &v0 = blockInfo->center.vertex;
            auto &color = v0.color;
            for (auto &edgeInfo: blockInfo->edges) {
                drawer.appendVertex(sf::Triangles, v0);
                for (auto &vertexInfo: edgeInfo->vertexes) {
                    auto v = vertexInfo->point.vertex;
                    v.color = color;
                    drawer.appendVertex(sf::Triangles, v);
                }
            }
        }
    }

    std::string Blocks::getHintLabelText() {
        return "Initialized blocks' information.";
    }

    std::string Blocks::save() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("blocks");
        CreateDependentDirectory(fp);
        std::ofstream ofs(fp, std::ios_base::binary);
        if (ofs.good()) {
            BlockInfo::SaveBlockInfosTo(ofs, _blockInfos);
            return "Blocks saved.";
        } else {
            return "Blocks saving failed.";
        }
    }

    std::string Blocks::load() {
        const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("blocks");
        std::ifstream ifs(fp, std::ios_base::binary);
        if (ifs.good()) {
            _blockInfos.clear();
            BlockInfo::LoadBlockInfosTo(ifs, _blockInfos);
            return "Blocks loaded.";
        } else {
            return "Blocks loading failed.";
        }
    }
}
