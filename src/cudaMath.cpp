#include "cudaMath.h"

namespace cm
{

size_t get_pow_2(size_t n)
{
	int count = 0;

	n = std::round(n);

	while(n != 1)
	{
		++count;
		n >>= 1;
	}

	return n <<= ++count;
}

size_t next_multiple(size_t n, size_t f)
{
	size_t mod = n % f;
	if(mod)
		n += f - mod;
	return n;
}

size_t factor_of(size_t n, size_t max)
{
	size_t out = (n < max)? n : max;
	while(n % out)
		--out;
	return out;
}

bool is_pow_2(size_t n)
{
	if(n == 0)
		return false;

	int count = 0;
	while(n > 1)
	{
		count += (int)(n & 1);
		if(count)
			return false;
		n >>= 1;
	}

	return true;
}

} // cm namespace