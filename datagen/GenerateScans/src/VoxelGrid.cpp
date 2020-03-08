
#include "stdafx.h"

#include "VoxelGrid.h"


void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {
				if (!m_sceneBoundsGrid.intersects(vec3f((float)i, (float)j, (float)k))) continue; // not in obb of scene

				//transform to current frame
				vec3f pf = worldToCamera * voxelToWorld(vec3i(i, j, k));

				//project into depth image
				vec3f p = skeletonToDepth(intrinsic, pf);
				vec3i pi = math::round(p);

				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					float d = depthImage(pi.x, pi.y);
					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {
						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);
						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);

							Voxel& v = (*this)(i, j, k);
							if (v.sdf == -std::numeric_limits<float>::infinity()) {
								v.sdf = sdf;
							}
							else {
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());
						}
					}
				}

			}
		}
	}
}
