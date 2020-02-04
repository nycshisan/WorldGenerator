//
// Created by nycsh on 2020/2/2.
//

#ifndef WORLDGENERATOR_FSWRAPPER_H
#define WORLDGENERATOR_FSWRAPPER_H

#include <string>

namespace wg {

    bool CreateDependentDirectory(const std::string &fp);

    bool ClearDirectory(const std::string &dir);

}
#endif //WORLDGENERATOR_FSWRAPPER_H
