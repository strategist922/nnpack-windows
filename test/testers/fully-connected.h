#pragma once

#include <cstddef>
#include <cstdlib>

#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>
#include <numeric>

#include <nnpack.h>
#include <reference.h>
#include <utils.h>

#include <gtest\gtest.h>

class FullyConnectedTester {
public:
	FullyConnectedTester() :
		iterations_(1),
		errorLimit_(1.0e-5),
		multithreading_(false),
		batchSize_(1),
		inputChannels_(1),
		outputChannels_(1)
	{
		
	}

	FullyConnectedTester(const FullyConnectedTester&) = delete;

	inline FullyConnectedTester(FullyConnectedTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		batchSize_(tester.batchSize_),
		inputChannels_(tester.inputChannels_),
		outputChannels_(tester.outputChannels_)
		
	{
	
	}

	FullyConnectedTester& operator=(const FullyConnectedTester&) = delete;

	~FullyConnectedTester() {
		
	}

	inline FullyConnectedTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline FullyConnectedTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline FullyConnectedTester& multithreading(bool multithreading) {
		this->multithreading_ = multithreading;
		
		return *this;
	}

	inline bool multithreading() const {
		return this->multithreading_;
	}

	inline FullyConnectedTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline FullyConnectedTester& inputChannels(size_t inputChannels) {
		this->inputChannels_ = inputChannels;
		return *this;
	}

	inline size_t inputChannels() const {
		return this->inputChannels_;
	}

	inline FullyConnectedTester& outputChannels(size_t outputChannels) {
		this->outputChannels_ = outputChannels;
		return *this;
	}

	inline size_t outputChannels() const {
		return this->outputChannels_;
	}

	void testOutput() const {
		const uint_fast32_t seed = uint_fast32_t(std::chrono::system_clock::now().time_since_epoch().count());
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> input(batchSize() * inputChannels());
		std::vector<float> kernel(outputChannels() * inputChannels());

		std::vector<float> output(batchSize() * outputChannels());
		std::vector<float> referenceOutput(batchSize() * outputChannels());

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());

			nnp_fully_connected_output_f32__reference(
				batchSize(), inputChannels(), outputChannels(),
				input.data(), kernel.data(), referenceOutput.data());

			enum nnp_status status = nnp_fully_connected_output(
				batchSize(), inputChannels(), outputChannels(),
				input.data(), kernel.data(), output.data());
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			EXPECT_LT(maxError, errorLimit());
		}
	}

	void testInferenceF32() const {
		ASSERT_EQ(1, batchSize());

		const uint_fast32_t seed = uint_fast32_t(std::chrono::system_clock::now().time_since_epoch().count());
		auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));

		std::vector<float> input(inputChannels());
		std::vector<float> kernel(outputChannels() * inputChannels());

		std::vector<float> output(outputChannels());
		std::vector<float> referenceOutput(outputChannels());

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());

			nnp_fully_connected_output_f32__reference(
				1, inputChannels(), outputChannels(),
				input.data(), kernel.data(), referenceOutput.data());

			enum nnp_status status = nnp_fully_connected_inference(
				inputChannels(), outputChannels(),
				input.data(), kernel.data(), output.data());
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			EXPECT_LT(maxError, errorLimit());
		}
	}

	

private:
	inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	size_t iterations_;
	float errorLimit_;
	bool multithreading_;

	size_t batchSize_;
	size_t inputChannels_;
	size_t outputChannels_;
};
