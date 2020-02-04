//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include <vector>

#include "impl/centers.h"
#include "impl/delaunay.h"
#include "impl/voronoi.h"
#include "impl/lloyd.h"
#include "impl/blocks.h"
#include "impl/finish.h"

#include "config.h"

namespace wg {

    class MainWindow;
    class Drawer;

    class Generator {
        Generator();

        void _setLabel(MainWindow &window);

        std::shared_ptr<Drawer> _drawer;

        std::vector<std::shared_ptr<GeneratorImpl>> impls;

        void _prepareConfigs();

    public:
        struct State {
            static const int Ready = -1;
            static const int Centers = 0;
            static const int DelaunayTriangles = 1;
            static const int VoronoiDiagram = 2;
            static const int LloydRelaxation = 3;
            static const int Blocks = 4;
            static const int Coast = 5;
            static const int Finish = 6;
        };

        int state = State::Ready;

        typedef std::vector<std::shared_ptr<GeneratorConfig>> Configs;
        Configs configs;

        static Generator &SharedInstance();

        void display(MainWindow &window);

        void next();

        void redo();

        void undo();

        std::string save();

        std::string load();

        void saveErrorData();

        static void NextButtonResponder(MainWindow &window);

        static void RedoButtonResponder(MainWindow &window);

        static void UndoButtonResponder(MainWindow &window);

        static void SaveButtonResponder(MainWindow &window);

        static void LoadButtonResponder(MainWindow &window);

        static void ConfigButtonResponder(MainWindow &window);
    };

}

#endif //WORLDGENERATOR_GENERATOR_H
