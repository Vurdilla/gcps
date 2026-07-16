#ifndef VC3_GENERAL_TIMER
#define VC3_GENERAL_TIMER


#include <string>
#include <iomanip>
//#include <dos>
#include <time.h>
#include "stringext.cu"

namespace vc3_general {

	enum timer_format
				{vtfHMS=1,      //HOURS:MINUTES:SECONDS
				 vtfMS=2,       //MINUTES:SECONDS
				 vtfDHMS=4,     //DAYS:HOURS:MINUTES:SECONDS
				 vtfS=3,        //SECONDS
				 vtfM=5,        //MINUTES
				 vtfH=6,        //HOURS
				 ctfUNKNOWN};

	class timer
	{
		protected:
		time_t start_sec;
		time_t stop_sec;
		bool running;


		public:
		timer() throw();
		void start() throw();
		void stop() throw();
		bool isrunning() const throw();
		tm get_ttime(timer_format format) const throw();
		std::string get_stime(timer_format format) const throw();
	};

} //namespace v3c_general


vc3_general::timer::timer() throw()
{
		running=false;
}

void vc3_general::timer::start() throw()
{
        start_sec=time(NULL);
        running=true;
}

void vc3_general::timer::stop() throw()
{
        if(running) stop_sec=time(NULL);
        running=false;
}

bool vc3_general::timer::isrunning() const throw()
{
        return running;
}

tm vc3_general::timer::get_ttime(timer_format format) const throw()
{
        tm d;
        d.tm_sec=stop_sec-start_sec;
        /*
                 vtfHMS=1,      //HOURS:MINUTES:SECONDS
                 vtfMS=2,       //MINUTES:SECONDS
                 vtfDHMS=4,     //DAYS:HOURS:MINUTES:SECONDS
                 vtfS=3,        //SECONDS
                 vtfM=5,       //MINUTES
                 vtfH=6,       //HOURS
                 ctfUNKNOWN};
        */
        if(format==vtfS) return d;
        d.tm_min=d.tm_sec/60;
        d.tm_sec%=60;
        if(format==vtfMS) return d;
        if(format==vtfM) {d.tm_sec=0; return d;}
        d.tm_hour=d.tm_min/60;
        d.tm_min%=60;
        if(format==vtfHMS) return d;
        if(format==vtfH) {d.tm_sec=d.tm_min=0; return d;}
        d.tm_mday=d.tm_hour/24;
        d.tm_hour%=24;
        if(format==vtfDHMS) return d;

        d.tm_sec=0;
        d.tm_min=0;
        d.tm_hour=0;
        d.tm_mday=0;
        return d;
}

std::string vc3_general::timer::get_stime(timer_format format) const throw()
{
        tm d;
        std::string s;
        d.tm_sec=stop_sec-start_sec;
        /*
                 vtfHMS=1,      //HOURS:MINUTES:SECONDS
                 vtfMS=2,       //MINUTES:SECONDS
                 vtfDHMS=4,     //DAYS:HOURS:MINUTES:SECONDS
                 vtfS=3,        //SECONDS
                 vtfM=5,       //MINUTES
                 vtfH=6,       //HOURS
                 ctfUNKNOWN};
		*/

        if(format==vtfS)
        {
				s=itoa(d.tm_sec)+" sec";
				return s;
        }
        d.tm_min=d.tm_sec/60;
        d.tm_sec%=60;
        if(format==vtfMS)
        {
                s=itoa(d.tm_min)+":"+itoa(d.tm_sec);
                return s;
        }
        if(format==vtfM)
        {
                s=itoa(d.tm_min)+" min";
                return s;
        }
        d.tm_hour=d.tm_min/60;
        d.tm_min%=60;
        if(format==vtfHMS)
        {
                s=itoa(d.tm_hour)+":"+itoa(d.tm_min)+":"+itoa(d.tm_sec);
                return s;
        }
        if(format==vtfH)
        {
                s=itoa(d.tm_hour)+" hrs";
                return s;
        }
        d.tm_mday=d.tm_hour/24;
        d.tm_hour%=24;
        if(format==vtfDHMS)
        {
                s=itoa(d.tm_mday)+":"+itoa(d.tm_hour)+":"+itoa(d.tm_min)+":"+itoa(d.tm_sec);
                return s;
        }
		return "";
}


#endif // VC3_GENERAL_TIMER
