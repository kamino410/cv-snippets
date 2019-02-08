
#ifdef _WIN32
#define UNITYCALLCONV __stdcall
#define UNITYEXPORT __declspec(dllexport)
#else
#define UNITYCALLCONV
#define UNITYEXPORT
#endif

extern "C" {
UNITYEXPORT int UNITYCALLCONV getNumber() { return 777; }
}
