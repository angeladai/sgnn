#pragma once

namespace CameraUtil {

	inline float gaussR(float sigma, float dist)
	{
		return std::exp(-(dist*dist) / (2.0f*sigma*sigma));
	}

	inline float linearR(float sigma, float dist)
	{
		return std::max(1.0f, std::min(0.0f, 1.0f - (dist*dist) / (2.0f*sigma*sigma)));
	}

	inline float gaussD(float sigma, int x, int y)
	{
		return std::exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
	}

	inline float gaussD(float sigma, int x)
	{
		return std::exp(-((x*x) / (2.0f*sigma*sigma)));
	}

	void bilateralFilter(const DepthImage32& input, float sigmaD, float sigmaR, DepthImage32& output)
	{
		if (output.getDimensions() != input.getDimensions())
			output.allocateSameSize(input);
		output.setPixels(-std::numeric_limits<float>::infinity());
		const int kernelRadius = (int)std::ceil(2.0*sigmaD);
#pragma omp parallel for
		for (int y = 0; y < (int)input.getHeight(); y++) {
			for (int x = 0; x < (int)input.getWidth(); x++) {

				float sum = 0.0f;
				float sumWeight = 0.0f;
				const float depthCenter = input(x, y);
				if (depthCenter != -std::numeric_limits<float>::infinity())
				{
					for (int m = (int)x - kernelRadius; m <= (int)x + kernelRadius; m++)
					{
						for (int n = (int)y - kernelRadius; n <= (int)y + kernelRadius; n++)
						{
							if (m >= 0 && n >= 0 && m < (int)input.getWidth() && n < (int)input.getHeight())
							{
								const float currentDepth = input(m, n);

								if (currentDepth != -std::numeric_limits<float>::infinity()) {
									const float weight = gaussD(sigmaD, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);

									sumWeight += weight;
									sum += weight*currentDepth;
								}
							}
						}
					}

					if (sumWeight > 0.0f) output(x, y) = sum / sumWeight;
					else output(x, y) = -std::numeric_limits<float>::infinity();
				}
				else output(x, y) = -std::numeric_limits<float>::infinity();
			} //x
		} //y
	}

	// thresh: max percentage of valid neighbors to be an edge
	void computeEdgeMask(const DepthImage32& input, float depthThresh, float thresh, int radius, BaseImage<unsigned char>& mask)
	{
		if (mask.getDimensions() != input.getDimensions())
			mask.allocate(input.getWidth(), input.getHeight());
		memset(mask.getData(), 0, sizeof(unsigned char)*mask.getNumPixels());
		const int size = (2 * radius + 1) * (2 * radius + 1);

		for (unsigned int y = 0; y < input.getHeight(); y++) {
			for (unsigned int x = 0; x < input.getWidth(); x++) {

				unsigned int count = 0;
				const float depthCenter = input(x, y);
				if (depthCenter != -std::numeric_limits<float>::infinity())
				{
					for (int m = (int)x - radius; m <= (int)x + radius; m++)
					{
						for (int n = (int)y - radius; n <= (int)y + radius; n++)
						{
							if (m >= 0 && n >= 0 && m < (int)input.getWidth() && n < (int)input.getHeight())
							{
								const float currentDepth = input(m, n);

								if (currentDepth != -std::numeric_limits<float>::infinity() && std::fabs(depthCenter - currentDepth) <= depthThresh) {
									count++;
								}
							}
						}
					}
					
					if ((float)count/(float)size <= thresh) mask(x, y) = 2; // edge
					else mask(x, y) = 1; //non-edge
				}
				else mask(x, y) = 0; //nothing
			} //x
		} //y
		
	}
}  // namespace CameraUtil