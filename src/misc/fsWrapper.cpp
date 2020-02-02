//
// Created by nycsh on 2020/2/2.
//

#include "fsWrapper.h"

#include <filesystem>

bool CreateDependentDirectory(const std::string &fp) {
    return std::filesystem::create_directories(std::filesystem::path(fp).remove_filename());
}

bool ClearDirectory(const std::string &dir) {
    return std::filesystem::remove_all(dir);
}
