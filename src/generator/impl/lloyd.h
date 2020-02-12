//
// Created by Nycshisan on 2018/4/4.
//

#ifndef WORLDGENERATOR_LLOYD_H
#define WORLDGENERATOR_LLOYD_H

#include "voronoi.h"

namespace wg {

    class LloydRelaxation : public GeneratorImpl {
    public:
        typedef VoronoiDiagram::Output Output;

    private:
        Output _relaxedVd;

    public:
        std::string getHintLabelText() override;

        void generate() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_LLOYD_H
