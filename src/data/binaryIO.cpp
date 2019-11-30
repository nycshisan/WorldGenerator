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
    // bool isCorner;
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
    write(ofs, info.isCorner);
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
}

void wg::BinaryIO::read(std::ifstream &ifs, bool& b) {
    unsigned char ucb;
    read(ifs, ucb);
    b = bool(ucb);
}

void wg::BinaryIO::read(std::ifstream &ifs, std::string &string, const size_t size) {
    char buf[size + 1]; buf[size] = 0;
    ifs.read(buf, size);
    string = buf;
}

void wg::BinaryIO::read(std::ifstream &ifs, wg::VertexInfo &info) {
    read(ifs, info.id);
    float x, y;
    read(ifs, x);
    read(ifs, y);
    info.point = Point(x, y);
    read(ifs, info.point.vertex.color.r);
    read(ifs, info.point.vertex.color.g);
    read(ifs, info.point.vertex.color.b);
    read(ifs, info.point.vertex.color.a);
    read(ifs, info.isCorner);
    read(ifs, info.blockIds);
    read(ifs, info.edgeIds);
}

void wg::BinaryIO::read(std::ifstream &ifs, wg::EdgeInfo &info) {
    read(ifs, info.id);
    read(ifs, info.isMargin);
    read(ifs, info.blockIds);
    read(ifs, info.vertexIds);
}

void wg::BinaryIO::read(std::ifstream &ifs, wg::BlockInfo &info) {
    read(ifs, info.id);
    float x, y;
    read(ifs, x);
    read(ifs, y);
    info.center = Point(x, y);
    read(ifs, info.center.vertex.color.r);
    read(ifs, info.center.vertex.color.g);
    read(ifs, info.center.vertex.color.b);
    read(ifs, info.center.vertex.color.a);
    read(ifs, info.edgeIds);
    read(ifs, info.vertexIds);
}
