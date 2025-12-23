#ifndef VC3_GENERAL_TIMERMSCUDA
#define VC3_GENERAL_TIMERMSCUDA


#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include "stringext.cu"

namespace vc3_general {

	class timer_ms_cuda
	{
		protected:
		std::chrono::high_resolution_clock::time_point tstart;
		std::chrono::high_resolution_clock::time_point tstop;
		bool running;


		public:
		timer_ms_cuda() throw();
		void start() throw();
		double lap() throw(); // returns last lap time in ms
		void stop() throw();
		bool isrunning() const throw();
		// Get time in units:
		// 0 - milliseconds
		// 1 - seconds
		// 2 - minutes
		// 3 - hours
		// 4 - days
		double get_dtime(int type=1) throw();
		// Get time in units:
		// 0 - milliseconds
		// 1 - seconds
		// 2 - minutes
		// 3 - hours
		// 4 - days
		std::string get_stime(int type=1, int precision=6) throw();
	};

} //namespace v3c_general


vc3_general::timer_ms_cuda::timer_ms_cuda() throw()
{
		running=false;
}

void vc3_general::timer_ms_cuda::start() throw()
{
        cudaDeviceSynchronize();
        tstart=std::chrono::high_resolution_clock::now();
        running=true;
}

double vc3_general::timer_ms_cuda::lap() throw()
{
    cudaDeviceSynchronize();
    tstop = std::chrono::high_resolution_clock::now();
    if (running == true)
    {
        std::chrono::duration<double, std::milli> time_span = tstop - tstart;
        tstart = tstop;
        return time_span.count(); // in milliseconds
    }
    else
    {
        tstart = tstop;
        running = true;
        return 0.0;
    }
}

void vc3_general::timer_ms_cuda::stop() throw()
{
        if(running)
        {
            cudaDeviceSynchronize();
            tstop=std::chrono::high_resolution_clock::now();
        }
        running=false;
}

bool vc3_general::timer_ms_cuda::isrunning() const throw()
{
        return running;
}

double vc3_general::timer_ms_cuda::get_dtime(int type) throw()
{
        if(running)
        {
            cudaDeviceSynchronize();
            tstop=std::chrono::high_resolution_clock::now();
        }
        std::chrono::duration<double, std::milli> time_span = tstop - tstart;
        double dt=time_span.count(); // in milliseconds
        switch(type)
        {
        case 4: // days
            return dt/86400000.00;
        case 3: // hours
            return dt/3600000.00;
        case 2: // minutes
            return dt/60000.00;
        case 1: // s
            return dt/1000.00;
        case 0: // ms
        default: // ms
            return dt;
        }
}

std::string vc3_general::timer_ms_cuda::get_stime(int type, int precision) throw()
{
        double dt=get_dtime(type);
        char c[6];
        std::string s;
        switch(type)
        {
        case 4: // days
            s=ftos_sp(dt,precision)+" days";
            break;
        case 3: // hours
            s=ftos_sp(dt,precision)+" hours";
            break;
        case 2: // minutes
            s=ftos_sp(dt,precision)+" min";
            break;
        case 1: // s
            s=ftos_sp(dt,precision)+" s";
            break;
        case 0:
        default:
            s=ftos_sp(dt,precision)+" ms";
        }
		return s;
}


#endif // VC3_GENERAL_TIMERMSCUDA
