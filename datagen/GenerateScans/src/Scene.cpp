
#include "stdafx.h"
#include "Scene.h"
#include "GlobalAppState.h"

void Scene::load(GraphicsDevice& g, const ScanInfo& scanInfo, bool bUseRenderedDepth, mat4f transform /*= mat4f::identity()*/) {
	clear();

	m_sensFiles = scanInfo.sensFiles;

	MeshDataf mesh = MeshIOf::loadFromFile(scanInfo.meshFile);
	if (!scanInfo.alnFile.empty()) {
		mat4f align = mat4f::identity();
		std::ifstream ifs(scanInfo.alnFile); std::string line;
		for (unsigned int i = 0; i < 3; i++) //read header
			std::getline(ifs, line);
		for (unsigned int r = 0; r < 4; r++) {
			for (unsigned int c = 0; c < 4; c++)
				ifs >> align(r, c);
		}
		ifs.close();
		mesh.applyTransform(align);
		mat4f translation = mat4f::translation(-mesh.computeBoundingBox().getMin());
		mesh.applyTransform(transform * translation);
		transform = transform * translation * align;
	}
	else {
		mesh.applyTransform(transform);
	}
	m_bb = mesh.computeBoundingBox();
	m_obb = OBB3f(mesh.m_Vertices, vec3f::eZ);

	m_bUseRenderedDepth = bUseRenderedDepth;
	if (bUseRenderedDepth) {
		m_sensDatas.resize(m_sensFiles.size(), nullptr);
		for (unsigned int i = 0; i < m_sensFiles.size(); i++) {
			m_sensDatas[i] = new SensorData(m_sensFiles[i]);
			for (unsigned int f = 0; f < m_sensDatas[i]->m_frames.size(); f++) {
				const mat4f frameTransform = m_sensDatas[i]->m_frames[f].getCameraToWorld();
				m_intrinsics.push_back(m_sensDatas[i]->m_calibrationDepth.m_intrinsic);
				m_extrinsics.push_back(transform * frameTransform);
				m_linearizedSensFrameIds.push_back(vec2ui(i, f));
			}
			m_sdDepthDims = vec2ui(m_sensDatas[i]->m_depthWidth, m_sensDatas[i]->m_depthHeight);
		}
	}
	else {
		m_sensDatas.resize(m_sensFiles.size(), nullptr);
		for (unsigned int i = 0; i < m_sensFiles.size(); i++) {
			m_sensDatas[i] = new SensorData(m_sensFiles[i]);
			for (unsigned int f = 0; f < m_sensDatas[i]->m_frames.size(); f++) {
				const mat4f frameTransform = m_sensDatas[i]->m_frames[f].getCameraToWorld();
				m_sensDatas[i]->m_frames[f].setCameraToWorld(transform * frameTransform);
				m_intrinsics.push_back(m_sensDatas[i]->m_calibrationDepth.m_intrinsic);
				m_extrinsics.push_back(transform * frameTransform);
				m_linearizedSensFrameIds.push_back(vec2ui(i, f));
			}
		}
	}

	{
		m_mesh.init(g, TriMeshf(mesh));

		// init rendering
		m_cbCamera.init(g);
		std::vector<DXGI_FORMAT> formats = {
			DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT
		};
		const unsigned int width = GlobalAppState::get().s_renderWidth;
		const unsigned int height = GlobalAppState::get().s_renderHeight;
		m_renderTarget.init(g.castD3D11(), width, height, formats, true);
	}
}

void Scene::updateRoom(GraphicsDevice& g, const ScanInfo& scanInfo, bool bUseRenderedDepth, mat4f transform /*= mat4f::identity()*/) {
	m_sensFiles = scanInfo.sensFiles;

	MeshDataf mesh = MeshIOf::loadFromFile(scanInfo.meshFile);
	if (!scanInfo.alnFile.empty()) {
		mat4f align = mat4f::identity();
		std::ifstream ifs(scanInfo.alnFile); std::string line;
		for (unsigned int i = 0; i < 3; i++) //read header
			std::getline(ifs, line);
		for (unsigned int r = 0; r < 4; r++) {
			for (unsigned int c = 0; c < 4; c++)
				ifs >> align(r, c);
		}
		ifs.close();
		mesh.applyTransform(align);
		mat4f translation = mat4f::translation(-mesh.computeBoundingBox().getMin());
		mesh.applyTransform(transform * translation);
		transform = transform * translation * align;
	}
	else {
		mesh.applyTransform(transform);
	}
	m_bb = mesh.computeBoundingBox();
	m_obb = OBB3f(mesh.m_Vertices, vec3f::eZ);

	m_bUseRenderedDepth = bUseRenderedDepth;
	{
		m_mesh.init(g, TriMeshf(mesh));
	}
}


bool Scene::renderDepthFrame(GraphicsDevice& g, unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth /*= 0.0f*/, float maxDepth /*= 12.0f*/) {
	const vec2ui& sensFrameId = m_linearizedSensFrameIds[idx];
	intrinsic = m_intrinsics[idx];
	extrinsic = m_extrinsics[idx];
	if (extrinsic[0] == -std::numeric_limits<float>::infinity()) return false;

	const unsigned int width = depth.getWidth();
	const unsigned int height = depth.getHeight();
	// adapt intrinsics
	intrinsic._m00 *= (float)width / (float)m_sdDepthDims[0];
	intrinsic._m11 *= (float)height / (float)m_sdDepthDims[1];
	intrinsic._m02 *= (float)(width - 1) / (float)(m_sdDepthDims[0] - 1);
	intrinsic._m12 *= (float)(height - 1) / (float)(m_sdDepthDims[1] - 1);
	const mat4f proj = Cameraf::visionToGraphicsProj(width, height, intrinsic(0, 0), intrinsic(1, 1), minDepth, maxDepth);
	const float fov = 2.0f * 180.0f / math::PIf * std::atan(0.5f * m_sdDepthDims[0] / intrinsic(0, 0));
	Cameraf cam = Cameraf(extrinsic, fov, (float)m_sdDepthDims[0] / (float)m_sdDepthDims[1], minDepth, maxDepth);

	ConstantBufferCamera cbCamera;
	cbCamera.worldViewProj = proj * cam.getView();

	m_cbCamera.updateAndBind(cbCamera, 0);
	g.castD3D11().getShaderManager().registerShader("shaders/drawAnnotations.hlsl", "drawAnnotations", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
	g.castD3D11().getShaderManager().bindShaders("drawAnnotations");

	m_renderTarget.clear();
	m_renderTarget.bind();
	m_mesh.render();
	m_renderTarget.unbind();

	m_renderTarget.captureDepthBuffer(depth);
	depth.setInvalidValue(-std::numeric_limits<float>::infinity());
	mat4f projToCamera = cam.getProj().getInverse();
	for (auto &p : depth) {
		vec3f posWorld = vec3f(-std::numeric_limits<float>::infinity());
		if (p.value != 0.0f && p.value != 1.0f) {
			vec3f posProj = vec3f(g.castD3D11().pixelToNDC(vec2i((int)p.x, (int)p.y), depth.getWidth(), depth.getHeight()), p.value);
			vec3f posCamera = projToCamera * posProj;
			if (posCamera.z >= 0.4f && posCamera.z <= 4.0f) {
				p.value = posCamera.z;
				posWorld = extrinsic * posCamera;
			}
			else {
				p.value = -std::numeric_limits<float>::infinity();
			}
		}
		else {
			p.value = -std::numeric_limits<float>::infinity();
		}
	} //depth pixels

	return true;
}

bool Scene::getDepthFrame(GraphicsDevice& g, unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth /*= 0.1f*/, float maxDepth /*= 12.0f*/) {
	if (m_bUseRenderedDepth)
		return renderDepthFrame(g, idx, depth, intrinsic, extrinsic, minDepth, maxDepth);
	else
		return getRawDepthFrame(idx, depth, intrinsic, extrinsic, minDepth, maxDepth);
}

bool Scene::getRawDepthFrame(unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth /*= 0.0f*/, float maxDepth /*= 12.0f*/) const {
	const vec2ui& sensFrameId = m_linearizedSensFrameIds[idx];
	const SensorData& sd = *m_sensDatas[sensFrameId.x];
	intrinsic = sd.m_calibrationDepth.m_intrinsic;
	extrinsic = sd.m_frames[sensFrameId.y].getCameraToWorld();
	if (extrinsic[0] == -std::numeric_limits<float>::infinity()) return false;

	const unsigned int newWidth = depth.getWidth();
	const unsigned int newHeight = depth.getHeight();
	float factorX = (float)(sd.m_depthWidth - 1) / (float)(newWidth - 1);
	float factorY = (float)(sd.m_depthHeight - 1) / (float)(newHeight - 1);

	//adapt intrinsics
	intrinsic._m00 *= (float)newWidth / (float)sd.m_depthWidth;
	intrinsic._m11 *= (float)newHeight / (float)sd.m_depthHeight;
	intrinsic._m02 *= (float)(newWidth - 1) / (float)(sd.m_depthWidth - 1);
	intrinsic._m12 *= (float)(newHeight - 1) / (float)(sd.m_depthHeight - 1);

	unsigned short* depthVals = sd.decompressDepthAlloc(sensFrameId.y);
	const float depthShift = 1.0f / sd.m_depthShift;
	for (unsigned int j = 0; j < newHeight; j++) {
		for (unsigned int i = 0; i < newWidth; i++) {
			const unsigned x = std::round((float)i * factorX);
			const unsigned y = std::round((float)j * factorY);
			const unsigned short d = depthVals[y*sd.m_depthWidth + x];
			if (d == 0) depth(i, j) = -std::numeric_limits<float>::infinity();
			else {
				float fd = depthShift * d;
				if (fd < minDepth || fd > maxDepth)
					depth(i, j) = -std::numeric_limits<float>::infinity();
				else
					depth(i, j) = fd;
			}
		}
	}
	std::free(depthVals);
	return true;
}