//
// Created by Nycshisan on 2019/11/29.
//

#ifndef WORLDGENERATOR_BINARYIO_H
#define WORLDGENERATOR_BINARYIO_H

#include <fstream>
#include <memory>
#include <type_traits>

#include "../misc/log.h"

namespace wg {

    class BlockInfo;

    namespace BinaryIO {
        typedef std::ofstream& OFS;

        template <typename T>
        void write(OFS ofs, std::enable_if<std::is_integral<T>::type, T> i) {
            LOG("integral");
            ofs.write((char*)&i, sizeof(T));
        }

        template <typename T>
        void write(OFS ofs, std::enable_if<std::is_same<T, size_t>::type, size_t> size) {
            LOG("size_t");
            ofs.write((char*)&size, sizeof(size_t));
        }

        void write(OFS ofs, const std::string& string);
        void write(OFS ofs, const BlockInfo& info);

        template <class T>
        void write(OFS ofs, const std::shared_ptr<T>& ptr) {
            const auto &ele = *ptr;
            write(ofs, ele);
        }

        template <class T>
        void write(OFS ofs, const std::vector<T>& vector) {
            write<size_t>(ofs, vector.size());
            for (const auto& ele : vector)
                write(ofs, ele);
        }
    }
}

#endif //WORLDGENERATOR_BINARYIO_H
