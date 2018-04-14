//
// Created by Nycshisan on 2018/4/8.
//

#ifndef WORLDGENERATOR_ERROR_H
#define WORLDGENERATOR_ERROR_H

#include <cassert>

void SaveErrorData();

#define assertWithSave(COND) do { if (!(COND)) { SaveErrorData(); assert(COND); } } while (false)

#endif //WORLDGENERATOR_ERROR_H
