//
// Created by nycsh on 2020/2/2.
//

#ifndef WORLDGENERATOR_COASTINFO_H
#define WORLDGENERATOR_COASTINFO_H

#include <memory>
#include <vector>

namespace wg {

    struct CoastInfo {
        enum class CoastType : int {
            Land = 1, Ocean = 2, Sea = 3, Unknown = 0
        };
        CoastType coastType = CoastType::Unknown;
        bool isContinentCenter = false;
        float height = -1;

        static void SaveCoastInfosTo(std::ofstream &ofs, const std::vector<CoastInfo> &infos);

        static void LoadCoastInfosTo(std::ifstream &ifs, std::vector<CoastInfo> &infos);
    };

}

#endif //WORLDGENERATOR_COASTINFO_H
