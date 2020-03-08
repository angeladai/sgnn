#pragma once 

#include "ScansDirectory.h"

class Visualizer : public ApplicationCallback
{
public:
	void init(ApplicationData &app);
	void render(ApplicationData &app);
	void keyDown(ApplicationData &app, UINT key);
	void keyPressed(ApplicationData &app, UINT key);
	void mouseDown(ApplicationData &app, MouseButtonType button);
	void mouseMove(ApplicationData &app);
	void mouseWheel(ApplicationData &app, int wheelDelta);
	void resize(ApplicationData &app);

	void process(ApplicationData& app, float scaleBounds = 1.0f);

private:

	void generateCompleteFrames(const Scene& scene, std::vector<unsigned int>& completeFrames) {
		completeFrames.clear();
		// matterport - filter out cameras not viewing the scene
		scene.computeTrajFramesInScene(completeFrames);
	}


	void generateIncompleteFramesMatterport(const Scene& scene, const std::vector<unsigned int>& completeFrames,
		float chanceDropFrame,  std::vector<unsigned int>& incompleteFrames) {
		incompleteFrames.clear();
		for (unsigned int f : completeFrames) {
			if (math::randomUniform(0.0f, 1.0f) > chanceDropFrame)
				incompleteFrames.push_back(f);
		}
	}

	ScansDirectory m_scans;

	Scene m_scene;

	D3D11Font m_font;
	FrameTimer m_timer;

	Cameraf m_camera;

	std::vector<std::vector<Cameraf>> m_recordedCameras;
	bool m_bEnableRecording;
	bool m_bEnableAutoRotate;
};