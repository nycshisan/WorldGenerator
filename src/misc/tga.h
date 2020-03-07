//
// Created by nycsh on 2020/3/3.
//

#ifndef WORLDGENERATOR_TGA_H
#define WORLDGENERATOR_TGA_H

#include <fstream>

namespace wg {

    class TGASaver {
        unsigned int _w, _h;

    public:
        TGASaver(unsigned int width, unsigned int height);

        bool save(const std::string &filename, unsigned char *data);
    };

}

#endif //WORLDGENERATOR_TGA_H
