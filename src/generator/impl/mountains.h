//
// Created by Nycshisan on 2020/2/12.
//

#ifndef WORLDGENERATOR_MOUNTAINS_H
#define WORLDGENERATOR_MOUNTAINS_H

#include "../impl.h"

namespace wg {

    class Mountains : public GeneratorImpl, private BlockHeightInfoDrawable {
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

#endif //WORLDGENERATOR_MOUNTAINS_H
