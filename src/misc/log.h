//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_LOG_H
#define WORLDGENERATOR_LOG_H

#include <iostream>
#include <string>

namespace wg {

    template<typename T>
    void _print(std::ostream &ostream, T &&arg) {
        ostream << std::forward<T>(arg) << " ";
    }

    template<typename... Args>
    void LOGIMPL(std::ostream &ostream, Args &&... args) {
        int _[] = {(_print(ostream, std::forward<Args>(args)), 0)...};
        ostream << std::endl;
    }

    template<typename... Args>
    void LOGOUT(Args &&... args) {
        LOGIMPL(std::cout, std::forward<Args>(args)...);
    }

}

#define LOG(args...) LOGOUT(std::string(__FILE__) + ":" + std::to_string(__LINE__), args)

#endif //WORLDGENERATOR_LOG_H
