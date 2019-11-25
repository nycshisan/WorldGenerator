//
// Created by Nycshisan on 2018/4/4.
//

#ifndef WORLDGENERATOR_LLOYD_H
#define WORLDGENERATOR_LLOYD_H

#include "voronoi.h"

namespace wg {

    class LloydRelaxation : public GeneratorImpl {
    public:
        typedef VoronoiDiagram::Output Input;
        typedef VoronoiDiagram::Output Output;

    private:
        Input _inputVd;
        Output _relaxedVd;

    public:
        std::string getHintLabelText() override;

        void input(void* inputData) override;

        void generate() override;

        void* output() override;

        void prepareVertexes(Drawer &drawer) override;
    };

}

#endif //WORLDGENERATOR_LLOYD_H
