
#include "stdafx.h"

#include "GlobalAppState.h"
#include "Fuser.h"
#include "MarchingCubes.h"
#include "CameraUtil.h"


Fuser::Fuser(ml::ApplicationData& _app) : m_app(_app)
{

}

Fuser::~Fuser()
{

}


void Fuser::fuse(const std::string& outputCompleteFile, const std::string& outputIncompleteFile,
	 Scene& scene, const std::vector<unsigned int>& completeFrames,
	const std::vector<unsigned int>& incompleteFrames, bool debugOut /*= false*/)
{
	const auto& gas = GlobalAppState::get();
	const float voxelSize = gas.s_voxelSize;
	const float depthMin = gas.s_renderNear;
	const float depthMax = gas.s_renderFar;
	const unsigned int scenePad = gas.s_scenePadding;
	const bool bSaveSparse = gas.s_bSaveSparse;
	const bool bFilterDepth = !gas.s_bUseRenderedDepth;
	const bool bGenerateSdfs = gas.s_bGenerateSdfs;
	const bool bGenerateKnown = gas.s_bGenerateKnown;
	const float saveSparseTruncFactor = 6.0f;
	const bool bGenerateComplete = !outputCompleteFile.empty();

	const unsigned int imageWidth = gas.s_renderWidth;
	const unsigned int imageHeight = gas.s_renderHeight;
	const unsigned int heightPad = gas.s_heightPad;

	const bbox3f bounds = scene.getBoundingBox();
	if (!bounds.isValid()) throw MLIB_EXCEPTION("invalid bounds for fuse");

	std::vector<DXGI_FORMAT> formats = { DXGI_FORMAT_R32G32B32A32_FLOAT };
	m_renderTarget.init(m_app.graphics.castD3D11(), imageWidth, imageHeight, formats);

	vec3ul voxelDim = math::round(bounds.getExtent() / voxelSize);
	//account for scene padding
	voxelDim += vec3ui(scenePad*2, scenePad*2, heightPad*2);
	const mat4f worldToGrid = mat4f::scale(1.0f / voxelSize) * mat4f::translation(-bounds.getMin() + vec3f(scenePad, scenePad, heightPad)*voxelSize);
	OBB3f sceneBoundsVoxels = worldToGrid * scene.getOBB();
	{ // add padding
		const vec3f center = sceneBoundsVoxels.getCenter();
		const vec3f axisX = sceneBoundsVoxels.getAxisX();
		const vec3f axisY = sceneBoundsVoxels.getAxisY();
		const vec3f axisZ = sceneBoundsVoxels.getAxisZ();
		sceneBoundsVoxels = OBB3f(center - 0.5f * (axisX + axisY + axisZ) - vec3f(scenePad, scenePad, heightPad), 
			(axisX.length() + scenePad*2) * axisX.getNormalized(), 
			(axisY.length() + scenePad * 2) * axisY.getNormalized(), 
			(axisZ.length() + heightPad * 2) * axisZ.getNormalized());
	}

	std::vector<unsigned int> initialFramesToScan = incompleteFrames;
	std::vector<unsigned int> restOfFramesToScan;
	{
		std::unordered_set<unsigned int> set(incompleteFrames.begin(), incompleteFrames.end());
		for (unsigned int f : completeFrames) {
			if (set.find(f) == set.end())
				restOfFramesToScan.push_back(f);
		}
	}

	VoxelGrid grid(voxelDim, worldToGrid, voxelSize, sceneBoundsVoxels, 0.4f, 4.0f);

	DepthImage32 rawDepth(imageWidth, imageHeight);
	DepthImage32 filtDepth(imageWidth, imageHeight);
	BaseImage<unsigned char> mask(imageWidth, imageHeight);
	mat4f intrinsic, extrinsic;
	for (unsigned int i = 0; i < initialFramesToScan.size(); i++) {
		bool bValid = scene.getDepthFrame(m_app.graphics, initialFramesToScan[i], rawDepth, intrinsic, extrinsic);
		if (bValid) {
			// filter the depth map
			if (bFilterDepth) CameraUtil::bilateralFilter(rawDepth, 2.0f, 0.1f, filtDepth);
			else std::swap(rawDepth, filtDepth);
			grid.integrate(intrinsic, extrinsic, filtDepth);
		}
		std::cout << "\r[ " << i << " | " << initialFramesToScan.size() << " ]";
	}
	if (bGenerateSdfs) { // save incomplete to file
		if (!outputIncompleteFile.empty()) {
			grid.saveToFile(outputIncompleteFile, bSaveSparse, saveSparseTruncFactor);
			grid.saveKnownToFile(util::removeExtensions(outputIncompleteFile) + ".knw");
		}

		if (debugOut) {
			MeshDataf obbMesh;
			for (const auto& e : sceneBoundsVoxels.getEdges())
				obbMesh.merge(Shapesf::cylinder(e.p0(), e.p1(), 0.1f, 10, 10, vec4f(1.0f, 0.0f, 0.0f, 1.0f)).computeMeshData());
			obbMesh.applyTransform(worldToGrid.getInverse());
			MeshIOf::saveToFile(util::removeExtensions(outputIncompleteFile) + "_BBOX-WORLD.ply", obbMesh);

			BinaryGrid3 bg = grid.toBinaryGridOccupied(0, grid.getVoxelSize());
			MeshIOf::saveToFile(util::removeExtensions(outputIncompleteFile) + "_INC-OCC.ply", TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).computeMeshData());
			VoxelGrid re(vec3ui(0, 0, 0), mat4f::identity(), 1.0f, OBB3f(), 0.1f, 10.0f);
			re.loadFromFile(outputIncompleteFile, bSaveSparse);

			const float eps = 0.00001f;
			if (std::fabs(voxelSize - re.getVoxelSize()) > eps) {
				std::cout << "error (incomplete) load/save voxel size " << voxelSize << " vs " << re.getVoxelSize() << std::endl;
				getchar();
			}
			const float thresh = saveSparseTruncFactor * grid.getVoxelSize();
			for (const auto& v : grid) {
				const Voxel& rv = re(v.x, v.y, v.z);
				if (std::fabs(rv.sdf) > thresh && std::fabs(v.value.sdf) > thresh) continue;
				if (std::fabs(rv.sdf - v.value.sdf) > eps) {
					std::cout << "error (incomplete) load/save voxel sdf " << v.value.sdf << " vs " << rv.sdf << std::endl;
					getchar();
				}
			}
			TriMeshf tri = TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 1.0f, 0.0f, 1.0f));
			tri.transform(mat4f::translation(bounds.getMin() - voxelSize * scenePad) * mat4f::scale(voxelSize));
			MeshIOf::saveToFile(util::removeExtensions(outputIncompleteFile) + "_OCC-INC-WORLD.ply", tri.computeMeshData());

			{ // marching cubes
				VoxelGrid copy(grid); copy.normalizeSDFs(); copy.setWorldToGrid(mat4f::identity());
				MeshDataf meshMC = MarchingCubes::doMC(copy, 10.0f, false);
				MeshIOf::saveToFile(util::removeExtensions(outputIncompleteFile) + "_INC-MC.ply", meshMC);
			}
		}
	}

	if (bGenerateComplete) { // scan complete
		for (unsigned int i = 0; i < restOfFramesToScan.size(); i++) {
			bool bValid = scene.getDepthFrame(m_app.graphics, restOfFramesToScan[i], rawDepth, intrinsic, extrinsic);
			if (bValid) {
				// filter the depth map
				if (bFilterDepth) CameraUtil::bilateralFilter(rawDepth, 2.0f, 0.1f, filtDepth);
				else std::swap(rawDepth, filtDepth);
				grid.integrate(intrinsic, extrinsic, filtDepth);
			}
			std::cout << "\r[ " << i << " | " << restOfFramesToScan.size() << " ]";
		}

		if (bGenerateSdfs) grid.saveToFile(outputCompleteFile, bSaveSparse, saveSparseTruncFactor);
		if (bGenerateKnown) grid.saveKnownToFile(util::removeExtensions(outputCompleteFile) + ".knw");

		if (debugOut) {
			BinaryGrid3 bg = grid.toBinaryGridOccupied(0, grid.getVoxelSize());
			MeshIOf::saveToFile(util::removeExtensions(outputCompleteFile) + "_OCC.ply", TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData());
			VoxelGrid re(vec3ui(0, 0, 0), mat4f::identity(), 1.0f, OBB3f(), 0.1f, 10.0f);
			re.loadFromFile(outputCompleteFile, bSaveSparse);

			const float eps = 0.00001f;
			if (std::fabs(voxelSize - re.getVoxelSize()) > eps) {
				std::cout << "error (complete) load/save voxel size " << voxelSize << " vs " << re.getVoxelSize() << std::endl;
				getchar();
			}
			const float thresh = saveSparseTruncFactor * grid.getVoxelSize();
			for (const auto& v : grid) {
				const Voxel& rv = re(v.x, v.y, v.z);
				if (std::fabs(rv.sdf) > thresh && std::fabs(v.value.sdf) > thresh) continue;
				if (std::fabs(rv.sdf - v.value.sdf) > eps) {
					std::cout << "error (complete) load/save voxel sdf " << v.value.sdf << " vs " << rv.sdf << std::endl;
					getchar();
				}
			}
			TriMeshf tri = TriMeshf(bg, mat4f::identity(), false, vec4f(0.0f, 0.0f, 1.0f, 1.0f));
			const unsigned int pad = scenePad;
			tri.transform(mat4f::translation(bounds.getMin() - voxelSize * pad) * mat4f::scale(voxelSize));
			MeshIOf::saveToFile(util::removeExtensions(outputCompleteFile) + "_OCC-WORLD.ply", tri.computeMeshData());

			{ // marching cubes
				VoxelGrid copy(grid); copy.normalizeSDFs(); copy.setWorldToGrid(mat4f::identity()); 
				MeshDataf meshMC = MarchingCubes::doMC(copy, 10.0f, false);
				MeshIOf::saveToFile(util::removeExtensions(outputCompleteFile) + "_MC.ply", meshMC);
			}
		}
	}
}