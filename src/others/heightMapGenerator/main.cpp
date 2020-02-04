//
// Created by nycsh on 2020/2/4.
//

#include "conf/conf.h"
#include "misc/misc.h"

int main() {
    auto width = CONF.getHeightMapWidth(), height = CONF.getHeightMapHeight();
    auto *heights = new float[height * width];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            heights[i * width + j] = wg::NoiseGenerator::PerlinNoise(float(i), float(j));
        }
    }

    const auto &fp = CONF.getOutputDirectory() + CONF.getHeightMapOutputPath();
    std::ofstream ofs(fp, std::ios_base::binary);

    ofs.write(reinterpret_cast<const char *>(heights), sizeof(float) * width * height);
    if (CONF.getInstallEnable()) {
        wg::CopyFile(fp, CONF.getInstallTarget() + CONF.getHeightMapOutputPath());
    }
}