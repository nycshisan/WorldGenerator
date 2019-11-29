//
// Created by Nycshisan on 2019/11/29.
//

#include "binaryIO.h"

#include "blockInfo.h"

void wg::BinaryIO::write(std::ofstream &ofs, const std::string& string) {
    ofs.write(string.c_str(), string.size());
}

void wg::BinaryIO::write(std::ofstream &ofs, const wg::BlockInfo &info) {
    // Binary data model in file:
    // unsigned int id;
    // float x, y;
    // unsigned char r, g, b, a;
//    write(ofs, info.id);
//    write(ofs, info.center.x);
//    write(ofs, info.center.y);
//    write(ofs, info.center.vertex.color.r);
}
