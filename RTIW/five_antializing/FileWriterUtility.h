#include <iostream>
#include <fstream>

std::ofstream fout("img5.ppm");

#define fp(a) fout << #a " = " << a << fflush
#define fpclose fout.close()

typedef void(*fendlManipulatorFunc)(std::ofstream&, int);
typedef std::ofstream& (*fflushManipulatorFunc)(std::ofstream&, int, int);

template <class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>& operator<<(std::basic_ostream<_CharT, _Traits>& fout, fendlManipulatorFunc m) {
    m(reinterpret_cast<std::ofstream&>(fout), 0);
    return fout;
}

template <class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>& operator<<(std::basic_ostream<_CharT, _Traits>& fout, fflushManipulatorFunc m) {
    m(reinterpret_cast<std::ofstream&>(fout), 0, 0);
    return fout;
}

void fendl(std::ofstream& fout, int) {
    fout.put('\n').flush();
    fout.close();
}

std::ofstream& fflush(std::ofstream& fout, int, int) {
    fout.put('\n').flush();
    return fout;
}


