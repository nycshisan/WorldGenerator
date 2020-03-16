//
// Created by nycsh on 2020/3/7.
//

#include "distField.h"

#include "blockEdges.h"

std::string wg::DistField::getHintLabelText() {
    return "Block edges distance field generated.";
}

void wg::DistField::generate() {
    delete[] _dfx;
    delete[] _dfy;

    _blockInfos = *(BlockEdges::Output*)_inputData;

    int size = CONF.getDistFieldSize();
    auto width = size, height = size;

    _t.create(width, height);
    _s.setTexture(_t);
    float scaleX = float(CONF.getUIMapWidth()) / float(width), scaleY = float(CONF.getUIMapHeight()) / float(width);
    _s.setScale(scaleX, scaleY);

    _dfx = new float[width * height];
    _dfy = new float[width * height];

    auto handle = CMJFAHandleAlloc(size);

    _initJFA(handle);

    CMJFACalculate(handle, _dfx, _dfy);
    _maxDist = CMJFAGetStat(handle, CMJFAStatType::MaxDist);

    auto rgba = new unsigned char[width * height * 4];

    CMJFAGenerateTexture(handle, rgba);

    _t.update(rgba);
    delete[] rgba;

    CMJFAHandleFree(handle);

    _outputData = (void*)&_blockInfos;
}

void wg::DistField::prepareVertexes(wg::Drawer &drawer) {
    drawer.addSprite(_s);
}

void wg::DistField::_initJFA(CMJFAHandle *handle) {
    std::unordered_set<std::shared_ptr<EdgeInfo>> edges;

    for (const auto &block: _blockInfos) {
        for (const auto &edge: block->edges) {
            edges.emplace(edge);
        }
    }

    float sx = (float)CONF.getDistFieldSize() / CONF.getMapWidth(),
          sy = (float)CONF.getDistFieldSize() / CONF.getMapHeight();

    Point a(888, 6999), b(5120, 1178);

    int stepNum = 100;
    float step = 1.f / (float)stepNum, crt = 0;
    while (crt < 1) {
        auto p = a * crt + b * (1 - crt);
        p.x *= sx; p.y *= sy;
        CMJFASetInitPoint(handle, p.x, p.y);
        crt += step;
    }
}

static std::string EDFPrefix = "EDFPrefix"; // NOLINT(cert-err58-cpp)
static std::string EDFSuffix = "EDFSuffix"; // NOLINT(cert-err58-cpp)

std::string wg::DistField::save() {
    const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("distField");
    CreateDependentDirectory(fp);
    std::ofstream ofs(fp, std::ios_base::binary);
    if (ofs.good()) {
        int size = CONF.getDistFieldSize();
        BinaryIO::write(ofs, size);
        BinaryIO::write(ofs, _maxDist);

        BinaryIO::write(ofs, EDFPrefix);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                int id = i * size + j;
                BinaryIO::write(ofs, _dfx[id]);
                BinaryIO::write(ofs, _dfy[id]);
            }
        }
        BinaryIO::write(ofs, EDFSuffix);

        return "Edge distance field saved.";
    } else {
        return "Edge distance field saving failed.";
    }
}

std::string wg::DistField::load() {
    const auto &fp = CONF.getOutputDirectory() + CONF.getModuleOutputPath("distField");
    std::ifstream ifs(fp, std::ios_base::binary);
    if (ifs.good()) {
        int textureSize;
        float maxDist;
        BinaryIO::read(ifs, textureSize);
        BinaryIO::read(ifs, maxDist);

        std::string s;
        BinaryIO::read(ifs, s, EDFPrefix.size());
        if (EDFPrefix != s) {
            LOG("Invalid EDF file prefix!");
        }

        auto rgba = new unsigned char[textureSize * textureSize * 4];

        for (int i = 0; i < textureSize; ++i) {
            for (int j = 0; j < textureSize; ++j) {
                float dx, dy;
                BinaryIO::read(ifs, dx);
                BinaryIO::read(ifs, dy);

                float dist = std::hypot(dx, dy);
                unsigned char color = int(dist / maxDist * 255);
                auto tb = (i * textureSize + j) * 4;
                rgba[tb] = rgba[tb + 1] = rgba[tb + 2] = color;
                rgba[tb + 3] = 255;
            }
        }

        BinaryIO::read(ifs, s, EDFSuffix.size());
        if (EDFSuffix != s) {
            LOG("Invalid EDF file prefix!");
        }

        _t.update(rgba);
        delete[] rgba;

        return "Edge distance field loaded.";
    } else {
        return "Edge distance field loading failed.";
    }
}
