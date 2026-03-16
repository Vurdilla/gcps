#include "simulator.cu"
#include "config.cu"
#include "../include/general/stringext.cu"
#include "../include/general/timer_ms_cuda.cu"
#include "../include/math/randgen/uniform_ran2.cu"
//#include "../include/general/timer_omp.cu"
//#include "../include/general/timer_omp_cuda.cu"

#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "omp.h"
#include <unistd.h>

int N_STREAMS = DEFAULT_STREAMS_PER_GPU;


int main(int argc, char* argv[])
{
#ifdef DEBUG0
    std::cout << "\nPROGRAM START\n";
    std::cout.flush();
#endif
    vc3_general::timer_ms_cuda timer_total;
    timer_total.start();

#ifdef DEBUG0
    std::cout << "\nREAD SETUP\n";
    std::cout.flush();
#endif
    vc3_phys::setupParameters setup;
    setup.read(argv[1]);
    setup.write("run_setup.ini");
#ifdef DEBUG0
    std::cout << "\nREAD ARGS\n";
    std::cout.flush();
#endif
    int cudaDeviceID = atoi(argv[2]);
    int nStreams = atoi(argv[3]);
    int randseed = atoi(argv[4]);
    std::string outfname = argv[5];

    double start, end;
    start = omp_get_wtime();
#ifdef DEBUG0
    std::cout << "\nCREATE KSNPSimulator\n";
    std::cout.flush();
#endif
    vc3_phys::KSNPSimulator simulator(setup, nStreams, randseed, cudaDeviceID);
#ifdef DEBUG0
    std::cout << "\nRUN KSNPSimulator\n";
    std::cout.flush();
#endif
    simulator.runSimulations();    
    end = omp_get_wtime();
    printf("\nrunSimulations wall time %f s\n", end - start);

#ifdef DEBUG0
    std::cout << "\nWRITE STAT\n";
    std::cout.flush();
#endif    
    simulator.writeStat(outfname);

    timer_total.stop();
    std::cout << "\n";
    std::cout << "--- Job time summary ---\n";
    std::cout << "Total wall clock time:\t" << timer_total.get_stime(1) << "\n";
    std::cout << "[ Did you expect this? ]\n";
#ifdef DEBUG0
    std::cout << "\nALL DONE\n";
    std::cout.flush();
#endif

    return 0;
}
