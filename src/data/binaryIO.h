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

        typedef std::ifstream& IFS;

        void read(IFS ifs, bool& b);
        void read(IFS ifs, std::string& string, size_t size);
        void read(IFS ifs, VertexInfo& info);
        void read(IFS ifs, EdgeInfo& info);
        void read(IFS ifs, BlockInfo& info);

        template <typename T>
        void_if<std::is_arithmetic<T>::value> read(IFS ifs, T& number) {
            ifs.read((char*)&number, sizeof(T));
        }

        template <class T>
        void read(IFS ifs, std::shared_ptr<T>& ptr) {
            ptr.reset(new T());
            read(ifs, *ptr);
        }

        template <class T>
        void read(IFS ifs, std::vector<T>& vector, size_t size) {
            vector.clear();
            for (size_t i = 0; i < size; ++i) {
                T ele;
                read(ifs, ele);
                vector.emplace_back(ele);
            }
        }

        template <class T>
        void read(IFS ifs, std::vector<T>& vector) {
            size_t size;
            read(ifs, size);
            read(ifs, vector, size);
        }
    }
}

#endif //WORLDGENERATOR_BINARYIO_H
