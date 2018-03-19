//
// Created by Nycshisan on 2018/3/15.
//

#include "voronoi.h"

void VoronoiDiagram::init(VoronoiDiagram::InputCenters centers, VoronoiDiagram::InputTris tris) {
    _centers = std::move(centers);
    _tris = tris;
}

void VoronoiDiagram::generate() {

}

VoronoiDiagram::Output VoronoiDiagram::output() {
    return 0;
}

void VoronoiDiagram::draw(Window &window) {

}