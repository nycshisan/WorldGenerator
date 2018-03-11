//
// Created by Nycshisan on 2018/3/8.
//

#ifndef WORLDGENERATOR_GENERATOR_H
#define WORLDGENERATOR_GENERATOR_H

#include "../graphics/window.h"

enum GeneratorState {
    Ready,
    PointsPined
};

class Generator {
    friend void NextButtonResponder(Window &window);

    GeneratorState _state = Ready;
public:
    static Generator &SharedInstance();
};

void SaveButtonResponder(Window &window);
void NextButtonResponder(Window &window);

#endif //WORLDGENERATOR_GENERATOR_H
