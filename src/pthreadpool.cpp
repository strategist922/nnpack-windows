#if defined(_MSC_VER) && defined(__cplusplus)
	#include <ppl.h>
#else
	#include <cassert>
	#include <cstdio>
	#include <limits>
	#include <string>
	#include <type_traits>
	#include <utility>
	#include <vector>
	#include <future>
	#include <thread>

	struct blocked_range 
	{
	public:
		typedef size_t const_iterator;

		blocked_range(const size_t& begin, const size_t& end) : begin_(begin), end_(end) {}
		blocked_range(const int& begin, const int& end) : begin_(begin), end_(end) {}

		const_iterator begin() const { return begin_; }
		const_iterator end() const { return end_; }

	private:
		const size_t begin_;
		const size_t end_;
	};

	template <typename Func>
	void xparallel_for(const size_t& begin, const size_t& end, const Func &f) 
	{
		blocked_range r(begin, end);
		f(r);
	}

	template <typename Func>
	void parallel_for(const size_t& begin, const size_t& end, const Func &f) 
	{
		assert(end >= begin);
		const size_t nthreads = std::thread::hardware_concurrency();
		size_t blockSize = (end - begin) / nthreads;
		if (blockSize * nthreads < end - begin) blockSize++;

		std::vector<std::future<void>> futures;

		size_t blockBegin = begin;
		size_t blockEnd = blockBegin + blockSize;

		if (blockEnd > end) blockEnd = end;

		for (size_t i = 0ull; i < nthreads; i++) 
		{
			futures.push_back(std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] 
			{
				f(blocked_range(blockBegin, blockEnd));
			})));

			blockBegin += blockSize;
			blockEnd = blockBegin + blockSize;

			if (blockBegin >= end) 
				break;

			if (blockEnd > end) 
				blockEnd = end;
		}

		for (auto &future : futures) 
			future.wait();
	}

	template <typename T, typename U>
	bool value_representation(U const &value) 
	{
		return static_cast<U>(static_cast<T>(value)) == value;
	}

	template <typename T, typename Func>
	inline void for_(bool parallelize, const size_t& begin, const T& end, const Func &f) 
	{
		static_assert(std::is_integral<T>::value, "end must be integral type");
		parallelize = parallelize && value_representation<size_t>(end);
		parallelize ? parallel_for(begin, end, f) : xparallel_for(begin, end, f);
	}

	template <typename T, typename Func>
	inline void for_i(bool parallelize, const T& size, const Func &f)
	{
		for_(parallelize, 0ull, size,[=](const blocked_range &r) 
		{
			for (int i = static_cast<int>(r.begin()); i < static_cast<int>(r.end()); i++) 
				f(i);
		});
	}

	template <typename T, typename Func>
	inline void for_i(const T& size, const Func &f)
	{
		for_i(true, size, f);
	}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <nnpack/pthreadpool.h>
#include <nnpack/utils.h>

#ifdef NOMINMAX
	static inline size_t min(size_t a, size_t b)
	{
		return a > b ? b : a;
	}
#endif // NOMINMAX

	void pthreadpool_compute_1d(
		pthreadpool_function_1d_t function,
		void* argument,
		const size_t range)
	{
#if defined(_MSC_VER) && defined(__cplusplus)
		concurrency::parallel_for(0ull, range, [=](size_t i)
		{
			function(argument, i);
		}, concurrency::static_partitioner());
#else
		for_i(range, [=](size_t i)
		{
			function(argument, i);
		});
#endif
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

		struct compute_1d_tiled_context context =
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
		struct compute_2d_context context =
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

		struct compute_2d_tiled_context context =
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
#ifdef __cplusplus
}
#endif
