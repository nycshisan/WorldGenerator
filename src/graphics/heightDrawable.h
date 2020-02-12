//
// Created by Nycshisan on 2020/2/12.
//

#ifndef WORLDGENERATOR_HEIGHTDRAWABLE_H
#define WORLDGENERATOR_HEIGHTDRAWABLE_H

#include "drawer.h"

namespace wg {

    class BlockHeightInfoDrawable {
        sf::Texture _t;
        sf::Sprite _s;

    public:
        void setTexture(const std::vector<std::vector<float>> &heights);

        void drawHeightsTo(Drawer &drawer, const std::vector<std::shared_ptr<BlockInfo>> &blockInfos);
    };

}

#endif //WORLDGENERATOR_HEIGHTDRAWABLE_H
