//
// Created by Nycshisan on 2019/11/29.
//

#ifndef WORLDGENERATOR_BINARYIO_H
#define WORLDGENERATOR_BINARYIO_H

#include <fstream>
#include <memory>
#include <type_traits>
#include <vector>
#include <set>
#include <unordered_set>

#include "../misc/log.h"

namespace wg {

    class VertexInfo;
    class EdgeInfo;
    class BlockInfo;

    namespace BinaryIO {
        template <bool>
        struct void_if_helper {};

        template <>
        struct void_if_helper<true> { typedef void VoidType; };

        template <bool V>
        using void_if = typename void_if_helper<V>::VoidType;

        typedef std::ofstream& OFS;

        template <typename T>
        void_if<std::is_arithmetic<T>::value> write(OFS ofs, T number) {
            ofs.write((char*)&number, sizeof(T));
        }

        void write(OFS ofs, bool b);
        void write(OFS ofs, const std::string& string);
        void write(OFS ofs, const VertexInfo& info);
        void write(OFS ofs, const EdgeInfo& info);
        void write(OFS ofs, const BlockInfo& info);

        template <class T>
        void write(OFS ofs, const std::shared_ptr<T>& ptr) {
            const auto &ele = *ptr;
            write(ofs, ele);
        }

        template <class T>
        struct is_single_iterable_container : std::false_type {};

        template <class T>
        struct is_single_iterable_container<std::vector<T>> : std::true_type {};
        template <class T>
        struct is_single_iterable_container<std::set<T>> : std::true_type {};
        template <class T>
        struct is_single_iterable_container<std::unordered_set<T>> : std::true_type {};

        template <class C>
        void_if<is_single_iterable_container<C>::value> write(OFS ofs, const C& vector) {
            write(ofs, vector.size());
            std::ifstream ifs;
            for (const auto& ele : vector)
                write(ofs, ele);
        }
    }
}

#endif //WORLDGENERATOR_BINARYIO_H
