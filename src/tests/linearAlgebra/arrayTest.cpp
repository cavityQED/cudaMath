#include "cudaMath.h"
#include "array.h"
#include "utilities.h"

typedef double dataType;
typedef cm::linear_algebra::Array<dataType>::shape_t	shape;
typedef cm::linear_algebra::Array<dataType>				array;
typedef boost::numeric::ublas::matrix<dataType>			matrix;

cm::Timer 							timer;
std::vector<std::shared_ptr<array>>	test_arrays;
std::vector<shape>	test_shapes{	{28, 1, 1, 1},
									{28, 1, 1, 28},
									{28, 28, 3, 1},
									{28, 28, 1, 32},
									{28, 28, 3, 32},
									{28, 28, 1, 128},
									{28, 28, 3, 128},
									{256, 256, 1, 1},
									{512, 512, 1, 1},
									{1024, 1024, 1, 1}};

size_t									size;
int										cur_time;
std::string								str_tmp;						
std::vector<std::vector<std::string>>	col_data;
std::vector<std::string>				shapes;
std::vector<std::string>				sizes;
std::vector<std::string>				times;
std::vector<std::string>				tpers;
std::vector<std::string>				results;
std::vector<std::string>				headers		{"Shape", "Size", "Time (ns)", "Time/Elem", "Result"};

void clear_vectors()
{
	col_data.clear();
	shapes.clear();
	sizes.clear();
	times.clear();
	tpers.clear();
	results.clear();
}

void matPrint(const matrix& mat)
{
	size_t row = 0;
	for(auto it1 = mat.begin1(); it1 != mat.end1(); it1++)
	{
		std::cout << "Row " << row << ":\t";
		for(auto it2 = it1.begin(); it2 != it1.end(); it2++)
			std::cout << *it2 << " ";
		std::cout << '\n';
		row++;
	}
}

std::shared_ptr<matrix> matAdd(const matrix& lhs, const matrix& rhs)
{
	auto out = std::make_shared<matrix>(lhs.size2(), lhs.size1());
	
	if(lhs.size1() * lhs.size2() != rhs.size1() * rhs.size2())
		return out;

	const	dataType* lhs_ptr = &lhs(0,0);
	const	dataType* rhs_ptr = &rhs(0,0);
			dataType* out_ptr = &(*out)(0,0);

	auto end = lhs_ptr + (lhs.size1() * lhs.size2());

	while(lhs_ptr != end)
	{
		*out_ptr = *lhs_ptr + *rhs_ptr;

		lhs_ptr++;
		rhs_ptr++;
		out_ptr++;
	}

	return out;
}
		
std::shared_ptr<matrix> matSub(const matrix& lhs, const matrix& rhs)
{
	auto out = std::make_shared<matrix>(lhs.size2(), lhs.size1());
	
	if(lhs.size1() * lhs.size2() != rhs.size1() * rhs.size2())
		return out;

	const	dataType* lhs_ptr = &lhs(0,0);
	const	dataType* rhs_ptr = &rhs(0,0);
			dataType* out_ptr = &(*out)(0,0);

	auto end = lhs_ptr + (lhs.size1() * lhs.size2());

	while(lhs_ptr != end)
	{
		*out_ptr = *lhs_ptr - *rhs_ptr;

		lhs_ptr++;
		rhs_ptr++;
		out_ptr++;
	}

	return out;
}

std::shared_ptr<matrix> matProd(const matrix& lhs, const matrix& rhs)
{
//	timer.start("boost matrix product");
	matrix tmp = prod(lhs, rhs);
//	timer.time("boost matrix product");

	auto lhs_ptr = &lhs(0,0);
	auto rhs_ptr = &rhs(0,0);
	auto tmp_ptr = &tmp(0,0);

	size_t nx = rhs.size2();
	size_t ny = lhs.size1();
	size_t ne = lhs.size2();

	size_t curx = 0;
	size_t cury = 0;

	dataType sum = 0;
//	timer.start("ptr matrix product");
	while(cury < ny)
	{
		curx = 0;
		while(curx < nx)
		{
			sum = 0;
			lhs_ptr = &lhs(0,0) + cury*ne;
			rhs_ptr = &rhs(0,0) + curx;

			for(size_t e = 0; e < ne; e++)
			{
				sum += *lhs_ptr * *rhs_ptr;
				lhs_ptr += 1;
				rhs_ptr += nx;
			}
			*tmp_ptr = sum;
			tmp_ptr++;
			curx++;
		}
		cury++;
	}
//	timer.time("ptr matrix product");


//	timer.start("make shared");
	auto out = std::make_shared<matrix>(tmp);
//	timer.time("make shared");

	return out;
}		

void makeArrays()
{
	std::cout << "\nMAKE ARRAYS:\n\n";

	test_arrays.clear();
	std::shared_ptr<array> arr;

	clear_vectors();

	for(auto s : test_shapes)
	{
		
		timer.start();
		arr = std::make_shared<array>(s);
		cur_time = timer.time_ns();
		
		size = arr->shape().size();
		sizes.push_back(std::to_string(size));
		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

		arr->shape().xyzb(str_tmp);
		shapes.push_back(str_tmp);

		if(cudaGetLastError() == cudaSuccess)
			results.push_back("Pass");
		else
			results.push_back("Fail");

		test_arrays.push_back(arr);
	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);

	print_table(headers, col_data);
}

void setData()
{
	cudaError_t						err;

	clear_vectors();

	std::cout << "\nSET DATA ONES:\n\n";
	for(auto arr : test_arrays)
	{
		arr->shape().xyzb(str_tmp);
		size = arr->shape().size();

		shapes.push_back(str_tmp);
		sizes.push_back(std::to_string(size));

		timer.start();
		err = arr->setData(1);
		cur_time = timer.time_ns();

		if(err == cudaSuccess)
			results.push_back("Pass");
		else
			results.push_back("Fail");

		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);

	print_table(headers, col_data);

	times.clear();
	tpers.clear();
	results.clear();
	col_data.clear();

	std::cout << "\nSET DATA RAND:\n\n";
	for(auto arr : test_arrays)
	{
		size = arr->shape().size();

		timer.start();
		err = arr->setRand();
		cur_time = timer.time_ns();
		
		if(err == cudaSuccess)
			results.push_back("Pass");
		else
			results.push_back("Fail");

		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));
	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);
	print_table(headers, col_data);
}

void add()
{
	std::shared_ptr<array>		arraySum;
	std::shared_ptr<matrix>		boostSum;

	clear_vectors();
	
	std::cout << "\nELEMENT ADD RAND:\n\n";
	for(auto arr : test_arrays)
	{
		arr->shape().xyzb(str_tmp);
		shapes.push_back(str_tmp);

		arr->setRand();
		timer.start();
		arraySum = array::element_add(*arr, *arr);
		cur_time = timer.time_ns();
		size = arr->shape().size();
		sizes.push_back(std::to_string(size));
		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

		arraySum->gpuRecv();
		boostSum = matAdd(arr->data(), arr->data());

		if(*arraySum == *boostSum)
			results.push_back("Pass");
		else
			results.push_back("Fail"); 
	}

	for(int i = 0; i < 256; i++)
		arraySum = array::element_add(*test_arrays.back(), *test_arrays.back());
	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);
	print_table(headers, col_data);
}

void add_constant(dataType c = 1)
{

	std::shared_ptr<array>	arrayTmp;
	std::shared_ptr<array>	arraySum;
	std::shared_ptr<matrix>	boostSum;

	clear_vectors();

	std::cout << "\nADD CONSTANT:\n\n";
	for(auto arr : test_arrays)
	{
		arr->shape().xyzb(str_tmp);
		shapes.push_back(str_tmp);

		arr->setRand();
		timer.start();
		arraySum = *arr + c;
		cur_time = timer.time_ns();
		size = arr->shape().size();
		sizes.push_back(std::to_string(size));
		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

		arraySum->gpuRecv();
		arrayTmp = std::make_shared<array>(arr->shape());
		arrayTmp->setData(c);
		boostSum = matAdd(arr->data(), arrayTmp->data());

		if(*arraySum == *boostSum)
			results.push_back("Pass");
		else
			results.push_back("Fail"); 
	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);
	print_table(headers, col_data);
}

void sub()
{
	std::shared_ptr<array>		arraySum;
	std::shared_ptr<matrix>		boostSum;

	clear_vectors();
	
	std::cout << "\nELEMENT SUB RAND:\n\n";
	for(auto arr : test_arrays)
	{
		arr->shape().xyzb(str_tmp);
		shapes.push_back(str_tmp);

		arr->setRand();
		timer.start();
		arraySum = array::element_add(*arr, *arr, -1);
		cur_time = timer.time_ns();
		size = arr->shape().size();
		sizes.push_back(std::to_string(size));
		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

		arraySum->gpuRecv();
		boostSum = matSub(arr->data(), arr->data());

		if(*arraySum == *boostSum)
			results.push_back("Pass");
		else
			results.push_back("Fail"); 
	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);
	print_table(headers, col_data);
}	
	
void prod()
{	
	clear_vectors();

	shape	out_shape	{32, 1, 1};
	shape	w_shape		{out_shape.height(),1,1,1};
	array	weights;

	std::cout << "\n2D PRODUCT RAND:\n\n";
	for(auto arr : test_arrays)
	{
		
		w_shape.b = arr->shape().rows();
		w_shape.pad(32);
		weights.resize(w_shape);
		weights.setRand(0, 1);
		
		out_shape.b = arr->shape().b;
		auto	out = std::make_shared<array>(out_shape);

		arr->shape().xyzb(str_tmp);
		shapes.push_back(str_tmp);

		arr->setRand(0,1);

		timer.start();
		out->product(weights, *arr);
		cur_time = timer.time_ns();

		size = arr->shape().size();
		sizes.push_back(std::to_string(size));
		times.push_back(std::to_string(cur_time));
		tpers.push_back(std::to_string((double)cur_time/size));

		out->gpuRecv();
		auto cmp = matProd(weights.data(), arr->data());

		if(*out == *cmp)
			results.push_back("Pass");
		else
			results.push_back("Fail");
	}

	col_data.push_back(shapes);
	col_data.push_back(sizes);
	col_data.push_back(times);
	col_data.push_back(tpers);
	col_data.push_back(results);
	print_table(headers, col_data);
}

void tr()
{
	std::cout << "Transpose Tests\n";
	std::cout << "---------------\n";

	std::cout << "Original:\n";

	auto arr = test_arrays[1];
	arr->setRand(-5, 5);
	arr->print();
	auto out = arr->transpose();
	out->gpuRecv();
	std::cout << "Transpose:\n";
	out->print();
}

void tableTest()
{
	std::vector<std::string> headers;
	headers.push_back("Header 1");
	headers.push_back("Head 2");

	std::vector<std::vector<std::string>>	col_data;

	col_data.push_back({"data", "more data"});
	col_data.push_back({"second data", "other"});

	print_table(headers, col_data);
}


int main(int argc, char const *argv[])
{
	cm::init();

	makeArrays();
	setData();
	add_constant(3.14);
	sub();
	prod();
	add();
//	tr();
	return 0;
}