//
// Created by Nycshisan on 2018/3/8.
//

#include "generator.h"

#include "../misc/log.h"
#include "../conf/conf.h"

void Generator::NextButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready:
            generator._blockCenters.input();
            generator._blockCenters.generate();
            break;
        case BlockCenters:
            generator._centers = generator._blockCenters.output();
            generator._delaunayTriangles.input(generator._centers);
            generator._delaunayTriangles.generate();
            break;
        case DelaunayTriangles:
            generator._tris = generator._delaunayTriangles.output();
            generator._voronoiDiagram.input(generator._centers, generator._tris);
            generator._voronoiDiagram.generate();
            break;
        case VoronoiDiagram:
            generator._vd = generator._voronoiDiagram.output();
            generator._lloydRelaxation.input(generator._vd);
            generator._lloydRelaxation.generate();
            break;
        case LloydRelaxation:
            generator._relaxedVd = generator._lloydRelaxation.output();
            generator._blocks.input(generator._relaxedVd);
            generator._blocks.generate();
            break;
        case Blocks:
            generator._blockInfos = generator._blocks.output();
            generator._coast.input(generator._blockInfos);
            generator._coast.generate();
            break;
        default:
            break;
    }
    generator._nextState();
    generator._setLabel(window);
}

void Generator::RedoButtonResponder(Window &window) {
    CONF.reload();
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case Ready: break;
        case BlockCenters:
            generator._blockCenters.generate(); break;
        case DelaunayTriangles:
            generator._delaunayTriangles.generate(); break;
        case VoronoiDiagram:
            generator._voronoiDiagram.generate(); break;
        case LloydRelaxation:
            generator._lloydRelaxation.generate(); break;
        case Blocks:
            generator._blocks.generate(); break;
        case Coast:
            generator._coast.generate(); break;
        default:
            LOGERR("Invalid generator state!");
    }
}
void Generator::UndoButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    generator._lastState();
    generator._setLabel(window);
}

void Generator::SaveButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case BlockCenters:
            generator._blockCenters.save();
            window.setHintLabel("Centers saved.");
            break;
        default:
            window.setHintLabel("Can't save.");
            generator.SaveErrorData();
            break;
    }
}

void Generator::LoadButtonResponder(Window &window) {
    Generator &generator = Generator::SharedInstance();
    switch (generator._state) {
        case BlockCenters:
            generator._blockCenters.load();
            window.setHintLabel("Centers loaded.");
            break;
        default:
            window.setHintLabel("Can't load.");
            break;
    }
}


Generator &Generator::SharedInstance() {
    static Generator instance;
    return instance;
}

void Generator::_nextState() {
    switch (_state) {
        case Ready:
            _state = BlockCenters; break;
        case BlockCenters:
            _state = DelaunayTriangles; break;
        case DelaunayTriangles:
            _state = VoronoiDiagram; break;
        case VoronoiDiagram:
            _state = LloydRelaxation; break;
        case LloydRelaxation:
            _state = Blocks; break;
        case Blocks:
            _state = Coast; break;
        default:
            break;
    }
}

void Generator::_lastState() {
    switch (_state) {
        case BlockCenters:
            _state = Ready; break;
        case DelaunayTriangles:
            _state = BlockCenters; break;
        case VoronoiDiagram:
            _state = DelaunayTriangles; break;
        case LloydRelaxation:
            _state = VoronoiDiagram; break;
        case Blocks:
            _state = LloydRelaxation; break;
        case Coast:
            _state = Blocks; break;
        default:
            break;
    }
}

void Generator::_setLabel(Window &window) {
    switch(_state) {
        case Ready:
            window.setHintLabel("Ready!"); break;
        case BlockCenters:
            window.setHintLabel("Generated block centers."); break;
        case DelaunayTriangles:
            window.setHintLabel("Generated Delaunay triangles."); break;
        case VoronoiDiagram:
            window.setHintLabel("Generated Voronoi diagram."); break;
        case LloydRelaxation:
            window.setHintLabel("Done Lloyd relaxation."); break;
        case Blocks:
            window.setHintLabel("Initialized blocks' information."); break;
        case Coast:
            window.setHintLabel("Generated the coast."); break;
        default:
            break;
    }
}

void Generator::display(Window &window) {
    if (!_drawer) {
        _drawer = std::shared_ptr<Drawer>::make_shared(&window);
    }

    switch (_state) {
        case BlockCenters:
            _blockCenters.draw(*_drawer); break;
        case DelaunayTriangles:
            _delaunayTriangles.draw(*_drawer); break;
        case VoronoiDiagram:
            _voronoiDiagram.draw(*_drawer); break;
        case LloydRelaxation:
            _lloydRelaxation.draw(*_drawer); break;
        case Blocks:
            _blocks.draw(*_drawer); break;
        case Coast:
            _coast.draw(*_drawer); break;
        default:
            break;
    }
}

void Generator::SaveErrorData() {
    _blockCenters.save();
}
