#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

namespace cm
{

	class Timer
	{

		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::time_point<Clock, std::chrono::nanoseconds> timePoint;

	public:

		Timer()
		{
			//m_startTime = Clock::now();
		}

		~Timer() {}

		void start()
		{
			m_startTime = Clock::now();
		}

		void start(const std::string& str)
		{
			m_startTime = Clock::now();
			std::cout << "\n\n\nTIMER START:\t" << str << '\n';
		}

		void time(const std::string& str)
		{
			m_stopTime = Clock::now();
			auto out = std::chrono::duration_cast<std::chrono::nanoseconds>(m_stopTime - m_startTime).count();
			std::cout << "\nTIMER STOP:\t" << str << "\n\tTime:\t" << out << " ns\n\n\n";
		}

		double time()
		{
			m_stopTime = Clock::now();
			return std::chrono::duration_cast<std::chrono::nanoseconds>(m_stopTime - m_startTime).count();
		}

		int time_ns()
		{
			return (int)time();
		}

	protected:

		timePoint	m_startTime;
		timePoint	m_stopTime;
	};

} // cm namespace


#endif