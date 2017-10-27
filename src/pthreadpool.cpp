#include <ppl.h>

#include <pthreadpool.h>
#include <utils.h>

void pthreadpool_compute_1d(
	pthreadpool_function_1d_t function,
	void* argument,
	const size_t range)
{
	concurrency::parallel_for(0ull, range, [=](size_t i) 
	{
		function(argument, i);
	}, concurrency::static_partitioner());
}

static void compute_1d_tiled(const struct compute_1d_tiled_context* context, const size_t linear_index)
{
	const size_t tile_index = linear_index;
	const size_t index = tile_index * context->tile;
	const size_t tile = min(context->tile, context->range - index);
	context->function(context->argument, index, tile);
}

void pthreadpool_compute_1d_tiled(
	pthreadpool_function_1d_tiled_t function,
	void* argument,
	const size_t range,
	const size_t tile)
{
	const size_t tile_range = divide_round_up(range, tile);
	compute_1d_tiled_context context = 
	{
		function,
		argument,
		range,
		tile
	};
	pthreadpool_compute_1d((pthreadpool_function_1d_t)compute_1d_tiled, &context, tile_range);
}

static void compute_2d(const struct compute_2d_context* context, const size_t linear_index)
{
	const struct fxdiv_divisor_size_t range_j = context->range_j;
	const struct fxdiv_result_size_t index = fxdiv_divide_size_t(linear_index, range_j);
	context->function(context->argument, index.quotient, index.remainder);
}

void pthreadpool_compute_2d(
	pthreadpool_function_2d_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j)
{
	compute_2d_context context  
	{
		function,
		argument,
		fxdiv_init_size_t(range_j)
	};

	pthreadpool_compute_1d((pthreadpool_function_1d_t)compute_2d, &context, range_i * range_j);
}

static void compute_2d_tiled(const struct compute_2d_tiled_context* context, const size_t linear_index)
{
	const struct fxdiv_divisor_size_t tile_range_j = context->tile_range_j;
	const struct fxdiv_result_size_t tile_index = fxdiv_divide_size_t(linear_index, tile_range_j);
	const size_t max_tile_i = context->tile_i;
	const size_t max_tile_j = context->tile_j;
	const size_t index_i = tile_index.quotient * max_tile_i;
	const size_t index_j = tile_index.remainder * max_tile_j;
	const size_t tile_i = min(max_tile_i, context->range_i - index_i);
	const size_t tile_j = min(max_tile_j, context->range_j - index_j);
	context->function(context->argument, index_i, index_j, tile_i, tile_j);
}

void pthreadpool_compute_2d_tiled(
	pthreadpool_function_2d_tiled_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j,
	const size_t tile_i,
	const size_t tile_j)
{
	const size_t tile_range_i = divide_round_up(range_i, tile_i);
	const size_t tile_range_j = divide_round_up(range_j, tile_j);

	compute_2d_tiled_context context = 
	{
		function,
		argument,
		fxdiv_init_size_t(tile_range_j),
		range_i,
		range_j,
		tile_i,
		tile_j
	};

	pthreadpool_compute_1d((pthreadpool_function_1d_t)compute_2d_tiled, &context, tile_range_i * tile_range_j);
}
