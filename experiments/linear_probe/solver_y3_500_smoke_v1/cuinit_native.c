/* No CUDA toolkit or Python required. The Driver API owns initialization. */
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    int (*init)(unsigned int) = (int (*)(unsigned int))dlsym(lib, "cuInit");
    if (!init) {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(lib);
        return 2;
    }
    int status = init(0u);
    printf("cuInit(0)=%d\n", status);
    dlclose(lib);
    return status ? 1 : 0;
}
