#include "graphics/graphics.h"

#include "cuda_modules/common/common.h"

int main() {
    if (CMIsOK()) {
        wg::MainWindow::MakeWindow()->play();
    } else {
        LOG("Cuda Modules DLL Error!");
    }


    return 0;
}