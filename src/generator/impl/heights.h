//
// Created by nycsh on 2020/2/10.
//

#ifndef WORLDGENERATOR_HEIGHTS_H
#define WORLDGENERATOR_HEIGHTS_H

#include "blocks.h"

#include "../config.h"

namespace wg {

    class BlockHeightInfoDrawable {
        sf::Texture _t;
        sf::Sprite _s;

    public:
        void createTexture(unsigned int width, unsigned int height);

        void setTexture(const std::vector<std::vector<float>> &heights);

        void drawHeightsTo(Drawer &drawer, const std::vector<std::shared_ptr<BlockInfo>> &blockInfos);
    };

    class Heights : public GeneratorImpl, private BlockHeightInfoDrawable {
    public:
        typedef Blocks::Output Input;
        typedef std::pair<Input, std::vector<std::vector<float>>> Output;

    private:
        Input _blockInfos;
        Output _blockHeightInfos;

    public:
        std::string getHintLabelText() override;

        void input(void* inputData) override;

        void generate() override;

        void* output() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_HEIGHTS_H
