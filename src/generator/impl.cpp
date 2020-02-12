//
// Created by nycsh on 2019/11/25.
//

#include "impl.h"

#include "generator.h"

namespace wg {

    std::string GeneratorImpl::save() {
        return "Can't save in this state.";
    }

    std::string GeneratorImpl::load() {
        return "Can't load in this state.";
    }

    void GeneratorImpl::getConfigs(Generator& generator) {}

    void GeneratorImpl::input(void *inputData) {
        _inputData = inputData;
    }

    void *GeneratorImpl::output() {
        return _outputData;
    }

}
