#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>

#include "timer.hpp"


void print_table(const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& col_data)
{
	if(col_data.size() != headers.size())
	{
		std::cout << "Table Size Mismatch\n";
		return;
	}

	std::vector<int> max_lens;
	for(auto h : headers)
		max_lens.push_back(h.length());
	for(int i = 0; i < max_lens.size(); i++)
	{
		for(auto str : col_data[i])
			max_lens[i] = (str.length() > max_lens[i])? str.length() : max_lens[i];
	}

	//Print the headers
	for(int col = 0; col < headers.size(); col++)
	{
		auto len = headers[col].length();
		auto sps = max_lens[col] - len;

		if(sps % 2)
		{
			sps++;
			max_lens[col]++;
		}

		//Print whitespace before header  
		for(int ws = 0; ws < sps/2; ws++)
			std::cout << " ";

		//Print the header
		std::cout << headers[col];

		//Print whitespace after header plus 2 spaces for column separation
		for(int ws = 0; ws < 2 + (sps/2); ws++)
			std::cout << " ";
	}

	std::cout << '\n';

	//Print lines to separate header row from rest of rows
	for(auto m : max_lens)
	{
		for(; m > 0; m--)
			std::cout << "-";
		std::cout << "  ";
	}

	std::cout << '\n';

	int rows = col_data[0].size();

	//Print the data
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < headers.size(); c++)
		{
			auto sps = max_lens[c] - col_data[c][r].length();
			for(int ws = 0; ws < sps; ws++)
				std::cout << " ";
			std::cout << col_data[c][r];
			std::cout << "  ";
		}
		std::cout << '\n';
	}

	std::cout << "\n\n";
}








namespace cm {	namespace utilities {







} // utilities namespace
} // cm namespace


#endif