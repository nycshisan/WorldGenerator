//
// Created by nycsh on 2020/2/10.
//

#ifndef WORLDGENERATOR_HEIGHTS_H
#define WORLDGENERATOR_HEIGHTS_H

#include "../impl.h"

namespace wg {

    class Heights : public GeneratorImpl, private BlockHeightInfoDrawable {
    public:
        typedef std::pair<std::vector<std::shared_ptr<BlockInfo>>, std::vector<std::vector<float>>> Output;

    private:
        Output _blockHeightInfos;

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_HEIGHTS_H
