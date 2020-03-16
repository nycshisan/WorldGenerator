#include "common/common.h" // This is the common header of cuda modules

bool TestCudaModules() {
    if (!CMIsOK()) {
        return false;
    }

    int srcSize = 16;
    auto src = new int[srcSize];
    for (int i = 0; i < srcSize; ++i) {
        src[i] = i;
    }
    auto r = CMCheckMemory(src, srcSize);
    for (int i = 0; i < srcSize; ++i) {
        if (src[i] != r[i]) {
            return false;
        }
    }
    CMFreeArray(r);

    return true;
}

#include "graphics/graphics.h"

#include "test/playground.h"

int main() {
    Playground();

    if (TestCudaModules()) {
        wg::MainWindow::MakeWindow()->play();
    } else {
        LOG("Cuda Modules DLL Error!");
    }

    return 0;
}