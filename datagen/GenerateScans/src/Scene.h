#pragma once

#include "GlobalAppState.h"
#include "ScansDirectory.h"
#include "Lighting.h"
#include "json.h"
#include "Segmentation.h"
#include "Aggregation.h"

class Scene
{
public:
	Scene() {
		m_graphics = nullptr;
	}

	~Scene() {
		clear();
	}

	void load(GraphicsDevice& g, const ScanInfo& scanInfo, bool bUseRenderedDepth, mat4f transform = mat4f::identity());
	void updateRoom(GraphicsDevice& g, const ScanInfo& scanInfo, bool bUseRenderedDepth, mat4f transform = mat4f::identity());

	const bbox3f& getBoundingBox() const {
		return m_bb;
	}
	const OBB3f& getOBB() const {
		return m_obb;
	}
	size_t getNumFrames() const {
		return m_linearizedSensFrameIds.size();
	}
	bool getDepthFrame(GraphicsDevice& g, unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth = 0.1f, float maxDepth = 12.0f);
	bool getRawDepthFrame(unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth = 0.1f, float maxDepth = 12.0f) const;
	bool renderDepthFrame(GraphicsDevice& g, unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth = 0.1f, float maxDepth = 12.0f);

	void randomizeLighting() {
		m_lighting.randomize();
	}

	const Lighting& getLighting() const {
		return m_lighting;
	}

	void setLighting(const Lighting& l) {
		m_lighting = l;
	}

	// trajectory frameIds <-> linearized sens frame idss
	void computeTrajFramesInScene(std::vector<unsigned int>& frameIds) const {
		frameIds.clear();
		if (!m_extrinsics.empty()) {
			for (unsigned int i = 0; i < m_extrinsics.size(); i++) {
				const mat4f& transform = m_extrinsics[i];
				if (m_obb.intersects(transform.getTranslation()))
					frameIds.push_back(i);
			}
		}
		else {
			std::vector<std::pair<unsigned int, float>> closestCameras; // if no cameras with center inside the room, use the 10 closest cameras...
			for (unsigned int i = 0; i < m_linearizedSensFrameIds.size(); i++) {
				const auto& sensFrameId = m_linearizedSensFrameIds[i];
				const mat4f& transform = m_sensDatas[sensFrameId.x]->m_frames[sensFrameId.y].getCameraToWorld();
				if (m_obb.intersects(transform.getTranslation()))
					frameIds.push_back(i);
				else
					closestCameras.push_back(std::make_pair(i, vec3f::dist(transform.getTranslation(), m_obb.getCenter())));
			}
			//const unsigned int minNumFrames = 10;
			const unsigned int minNumFrames = 30;
			if (true || frameIds.size() < minNumFrames) {
				std::sort(closestCameras.begin(), closestCameras.end(), [](const std::pair<unsigned int, float>& a, const std::pair<unsigned int, float>& b) {
					return a.second < b.second;
				});
				for (unsigned int i = 0; i < std::min((unsigned int)closestCameras.size(), minNumFrames); i++)
					frameIds.push_back(closestCameras[i].first);
			}
		}
	}

private:
	void clear() {
		m_sensFiles.clear();
		for (unsigned int i = 0; i < m_sensDatas.size(); i++)
			SAFE_DELETE(m_sensDatas[i]);
		m_sensDatas.clear();
		m_linearizedSensFrameIds.clear();
		m_bb.reset();
		m_obb.setInvalid();
	}

	static MeshDataf makeCamerasMesh(const std::vector<mat4f>& cameras, unsigned int skip = 1, const vec4f& eyeColor = vec4f(0.0f, 1.0f, 0.0f, 1.0f),
		const vec4f& lookColor = vec4f(1.0f, 0.0f, 0.0f, 1.0f), const vec4f& upColor = vec4f(0.0f, 0.0f, 1.0f, 1.0f)) {
		MeshDataf camMesh;
		for (unsigned int c = 0; c < cameras.size(); c += skip) {
			const mat4f& cam = cameras[c];
			const vec3f eye = cam.getTranslation();
			const vec3f look = cam.getRotation() * -vec3f::eZ;
			const vec3f up = cam.getRotation() * vec3f::eY;
			camMesh.merge(Shapesf::cylinder(eye, eye + 0.2f * look, 0.1f, 10, 10, lookColor).computeMeshData());
			camMesh.merge(Shapesf::cylinder(eye, eye + 0.2f * up, 0.1f, 10, 10, upColor).computeMeshData());
			camMesh.merge(Shapesf::sphere(0.1f, eye, 10, 10, eyeColor).computeMeshData());
		}
		return camMesh;
	}

	struct ConstantBufferCamera {
		mat4f worldViewProj;
		mat4f world;
		vec4f eye;
	};

	struct ConstantBufferMaterial {
		vec4f ambient;
		vec4f diffuse;
		vec4f specular;
		float shiny;
		vec3f dummy;
	};


	GraphicsDevice* m_graphics;

	D3D11ShaderManager m_shaders;

	std::vector<vec2ui> m_linearizedSensFrameIds;
	std::vector<std::string> m_sensFiles;
	std::vector<SensorData*> m_sensDatas;
	std::vector<mat4f> m_intrinsics, m_extrinsics;
	vec2ui m_sdDepthDims;
	bool m_bUseRenderedDepth;
	bbox3f m_bb;
	OBB3f m_obb;
	D3D11TriMesh m_mesh;

	D3D11ConstantBuffer<ConstantBufferCamera>	m_cbCamera;
	D3D11ConstantBuffer<ConstantBufferMaterial> m_cbMaterial;
	Lighting m_lighting;

	D3D11RenderTarget m_renderTarget;
};

