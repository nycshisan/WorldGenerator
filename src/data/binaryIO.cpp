//
// Created by Nycshisan on 2019/11/29.
//

#include "binaryIO.h"

#include "blockInfo.h"

void wg::BinaryIO::write(std::ofstream &ofs, bool b) {
    auto ucb = (unsigned char)(b);
    write(ofs, ucb);
}

void wg::BinaryIO::write(std::ofstream &ofs, const std::string& string) {
    ofs.write(string.c_str(), string.size());
}

void wg::BinaryIO::write(std::ofstream &ofs, const wg::VertexInfo &info) {
    // ----------------------------------------------
    // Binary data model in file:
    //
    // unsigned int id;
    // float point.(x, y);
    // unsigned char point.color.(r, g, b, a);
    // vector<unsigned int> relatedBlocks;
    // vector<unsigned int> relatedEdges;
    // ----------------------------------------------
    write(ofs, info.id);
    write(ofs, info.point.x);
    write(ofs, info.point.y);
    write(ofs, info.point.vertex.color.r);
    write(ofs, info.point.vertex.color.g);
    write(ofs, info.point.vertex.color.b);
    write(ofs, info.point.vertex.color.a);
    std::vector<unsigned int> ids;
    for (const auto &ptr : info.relatedBlocks) {
        ids.emplace_back(ptr.lock()->id);
    }
    write(ofs, ids);
    ids.clear();
    for (const auto &weakPtr : info.relatedEdges) {
        ids.emplace_back(weakPtr.lock()->id);
    }
    write(ofs, ids);
}

void wg::BinaryIO::write(std::ofstream &ofs, const wg::EdgeInfo &info) {
    // ----------------------------------------------
    // Binary data model in file:
    //
    // unsigned int id;
    // bool isMargin;
    // vector<unsigned int> relatedBlocks;
    // vector<unsigned int> vertexes;
    // ----------------------------------------------
    write(ofs, info.id);
    write(ofs, info.isMargin);
    std::vector<unsigned int> ids;
    for (const auto &ptr : info.relatedBlocks) {
        ids.emplace_back(ptr.lock()->id);
    }
    write(ofs, ids);
    ids.clear();
    for (const auto &ptr : info.vertexes) {
        ids.emplace_back(ptr->id);
    }
    write(ofs, ids);
}

void wg::BinaryIO::write(std::ofstream &ofs, const wg::BlockInfo &info) {
    // ----------------------------------------------
    // Binary data model in file:
    //
    // unsigned int id;
    // float center.(x, y);
    // unsigned char center.color.(r, g, b, a);
    // vector<unsigned int> edges;
    // vector<unsigned int> vertexes;
    // ----------------------------------------------
    write(ofs, info.id);
    write(ofs, info.center.x);
    write(ofs, info.center.y);
    write(ofs, info.center.vertex.color.r);
    write(ofs, info.center.vertex.color.g);
    write(ofs, info.center.vertex.color.b);
    write(ofs, info.center.vertex.color.a);
    std::vector<unsigned int> ids;
    for (const auto &ptr : info.edges) {
        ids.emplace_back(ptr->id);
    }
    write(ofs, ids);
    ids.clear();
    for (const auto &weakPtr : info.vertexes) {
        ids.emplace_back(weakPtr.lock()->id);
    }
    write(ofs, ids);
    write(ofs, int(info.coastType));
    write(ofs, info.isContinentCenter);
}
