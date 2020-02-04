//
// Created by nycsh on 2020/2/2.
//

#include "coastInfo.h"

#include <cassert>

#include "binaryIO.h"

namespace wg {

    static std::string CoastInfoVerifyHead = "WGCIHD"; // NOLINT(cert-err58-cpp)
    static std::string CoastInfoVerifyTail = "WGCITL"; // NOLINT(cert-err58-cpp)

    void CoastInfo::SaveCoastInfosTo(std::ofstream &ofs, const std::vector<CoastInfo> &infos) {
        using namespace BinaryIO;

        // save the information on each vertex
        write(ofs, CoastInfoVerifyHead);
        write(ofs, infos.size());
        for (const auto &ele : infos) {
            write(ofs, (int)ele.coastType);
            write(ofs, ele.isContinentCenter);
            write(ofs, ele.height);
        }

        // save the height map

        write(ofs, CoastInfoVerifyTail);
    }

    void CoastInfo::LoadCoastInfosTo(std::ifstream &ifs, std::vector<CoastInfo> &infos) {
        using namespace BinaryIO;

        std::string head;
        read(ifs, head, CoastInfoVerifyHead.size());
        assert(head == CoastInfoVerifyHead);

        size_t blockNum;
        read(ifs, blockNum);
        infos.resize(blockNum);

        for (int i = 0; i < blockNum; ++i) {
            int coastType;
            read(ifs, coastType);
            infos[i].coastType = static_cast<CoastType>(coastType);
            read(ifs, infos[i].isContinentCenter);
            read(ifs, infos[i].height);
        }

        std::string tail;
        read(ifs, tail, CoastInfoVerifyTail.size());
        assert(tail == CoastInfoVerifyTail);
    }

}