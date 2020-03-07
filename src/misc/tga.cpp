//
// Created by nycsh on 2020/3/3.
//

#include "tga.h"

namespace wg {

    TGASaver::TGASaver(unsigned int width, unsigned int height) : _w(width), _h(height) {}

    bool TGASaver::save(const std::string &filename, unsigned char *data) {
        std::ofstream tgafile(filename, std::ios::binary);
        if (!tgafile) return false;

        // The image header
        unsigned char header[18] = {};
        header[2] = 2;
        header[12] = _w & 0xFF;
        header[13] = (_w >> 8) & 0xFF;
        header[14] = _h & 0xFF;
        header[15] = (_h >> 8) & 0xFF;
        header[16] = 32;  // bits per pixel

        tgafile.write((const char*)header, 18);

        // The image data is stored bottom-to-top, left-to-right
        for (int y = _h - 1; y >= 0; y--) {
            for (int x = 0; x < _w; x++) {
                tgafile.put((char)data[((y * _w) + x) * 4 + 2]);
                tgafile.put((char)data[((y * _w) + x) * 4 + 1]);
                tgafile.put((char)data[((y * _w) + x) * 4 + 0]);
                tgafile.put((char)data[((y * _w) + x) * 4 + 3]);
            }
        }

        static const char footer[ 26 ] =
                "\0\0\0\0"  // no extension area
                "\0\0\0\0"  // no developer directory
                "TRUEVISION-XFILE"  // yep, this is a TGA file
                ".";
        tgafile.write( footer, 26 );

        tgafile.close();
        return true;
    }

}