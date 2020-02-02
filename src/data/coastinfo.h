//
// Created by nycsh on 2020/2/2.
//

#ifndef WORLDGENERATOR_COASTINFO_H
#define WORLDGENERATOR_COASTINFO_H

struct CoastInfo {
    enum class CoastType : int {
        Land = 1, Ocean = 2, Sea = 3, Unknown = 0
    };
    CoastType coastType = CoastType::Unknown;
    bool isContinentCenter = false;
};

#endif //WORLDGENERATOR_COASTINFO_H
