
#include "stdafx.h"

#include "GlobalAppState.h"
#include "Fuser.h"
#include "MarchingCubes.h"

#include "omp.h"

void Visualizer::init(ApplicationData& app)
{
	const auto& gas = GlobalAppState::get();

	const std::string sceneFileList = gas.s_sceneFileList;
	m_scans.loadMatterport(gas.s_scanPath, gas.s_scanMeshPath, sceneFileList, gas.s_maxNumSens);

	m_font.init(app.graphics, "Calibri");
	if (!sceneFileList.empty()) {
		process(app);
		exit(0);
	}
	m_bEnableRecording = false;
	m_bEnableAutoRotate = false;
}

void Visualizer::process(ApplicationData& app, float scaleBounds /*= 1.0f*/)
{
	const auto& gas = GlobalAppState::get();
	const bool debugOut = gas.s_bDebugVis;
	const std::string scanPath = gas.s_scanPath;
	const std::string scanMeshPath = gas.s_scanMeshPath;
	const std::string outputCompletePath = gas.s_outputCompletePath;
	const std::string outputIncompletePath = gas.s_outputIncompletePath;
	const std::string incompleteFramePath = gas.s_incompleteFramePath;
	const bool bGenerateSdfs = gas.s_bGenerateSdfs;
	const bool bGenerateKnown = gas.s_bGenerateKnown;
	const bool bUseRenderedDepth = gas.s_bUseRenderedDepth;
	const float chanceDropFrames = gas.s_chanceDropFrames;
	const float percentDropFrames = 0.25f;
	const unsigned int dropKeep = 100;

	if (!util::directoryExists(incompleteFramePath))
		util::makeDirectory(incompleteFramePath);

	const auto& scans = m_scans.getScans();
	std::unordered_map<std::string, std::vector<ScanInfo>> scenesToScans;
	for (int i = 0; i < (int)scans.size(); i++) {
		const auto& scanInfo = scans[i];
		const auto scene = util::splitOnFirst(scanInfo.sceneName, "_room").first;
		auto it = scenesToScans.find(scene);
		if (it == scenesToScans.end()) {
			scenesToScans[scene] = std::vector<ScanInfo>(1, scanInfo);
		}
		else {
			it->second.push_back(scanInfo);
		}
	}
	std::cout << "processing " << scans.size() << " scan | " << scenesToScans.size() << " scenes..." << std::endl;

	const bool bGenerateComplete = !outputCompletePath.empty();
	if (bGenerateComplete && !util::directoryExists(outputCompletePath)) util::makeDirectory(outputCompletePath);
	if (!util::directoryExists(outputIncompletePath)) util::makeDirectory(outputIncompletePath);

	const int numThreads = 1; 
	std::vector<Scene*> scenes(numThreads, NULL);
	std::vector<Fuser*> fusers(numThreads, NULL);
	for (unsigned int i = 0; i < numThreads; i++) {
		scenes[i] = new Scene;
		fusers[i] = new Fuser(app);
	}
	unsigned int _idx = 0;
	for (const auto& sceneInfo : scenesToScans) {
		std::cout << "\r(" << ++_idx << " | " << scenesToScans.size() << ")";
		Scene scene;
		scene.load(app.graphics, sceneInfo.second.front(), false);

		for (int i = 0; i < (int)sceneInfo.second.size(); i++) {
			const auto& scanInfo = sceneInfo.second[i];
			std::cout << "\r(" << _idx << " | " << scenesToScans.size() << ") [ " << (i + 1) << " | " << sceneInfo.second.size() << " ] " << scanInfo.sceneName;

			int thread = 0;
			if (true) { // skip if already exists
				const std::string outCompleteFile = outputCompletePath + "/" + scanInfo.sceneName + "__0__.sdf";
				const std::string outIncompleteFile = outputIncompletePath + "/" + scanInfo.sceneName + "__0__.sdf";
				if ((!bGenerateSdfs || ((!bGenerateComplete || util::fileExists(outCompleteFile)) && util::fileExists(outIncompleteFile))) &&
					(!bGenerateKnown || !bGenerateComplete || util::fileExists(outputCompletePath + "/" + scanInfo.sceneName + "__0__.knw"))) {
					if (i % 20 == 0)
						std::cout << "\r(" << _idx << " | " << scenesToScans.size() << ") [ " << (i + 1) << " | " << sceneInfo.second.size() << " ] (skip) " << scanInfo.sceneName << std::endl;
					continue;
				}
			}
			try {
				scene.updateRoom(app.graphics, scanInfo, bUseRenderedDepth);

				//generate complete/incomplete traj
				std::vector<unsigned int> completeFrames, incompleteFrames;
				{ // complete trajectory
					generateCompleteFrames(scene, completeFrames);
				}
				if (completeFrames.empty()) {
					std::cout << "\r[ " << (i + 1) << " | " << scanInfo.sceneName.size() << " ] (no frames) " << scanInfo.sceneName << std::endl;
					continue;
				}
				{ // incomplete trajectory
					const std::string incompleteFrameFile = incompleteFramePath + "/" + scanInfo.sceneName + "__0__.txt";
					if (!incompleteFramePath.empty() && util::fileExists(incompleteFrameFile)) {
						std::ifstream ifs(incompleteFrameFile);
						if (!ifs.good()) throw MLIB_EXCEPTION("failed to open incomplete frame file: " + incompleteFramePath + "/" + scanInfo.sceneName + ".txt");
						std::string line;
						while (std::getline(ifs, line))
							incompleteFrames.push_back(util::convertTo<unsigned int>(line));
					}
					else {
						generateIncompleteFramesMatterport(scene, completeFrames, chanceDropFrames, incompleteFrames);
						if (!incompleteFrames.empty()) {
							if (!util::directoryExists(incompleteFramePath)) util::makeDirectory(incompleteFramePath);
							std::ofstream ofs(incompleteFrameFile);
							for (const auto f : incompleteFrames) ofs << f << std::endl;
						}
					}
				}
				//fuse to sdf
				std::cout << "\r(" << _idx << " | " << scenesToScans.size() << ") [ " << (i + 1) << " | " << sceneInfo.second.size() << " ] " << scanInfo.sceneName << ": fusing " << completeFrames.size() << " complete | " << incompleteFrames.size() << " incomplete... ";
				Fuser& fuser = *fusers[thread];
				const std::string outCompleteFile = bGenerateComplete ? outputCompletePath + "/" + scanInfo.sceneName + "__0__.sdf" : "";
				const std::string outIncompleteFile = outputIncompletePath.empty() ? "" : outputIncompletePath + "/" + scanInfo.sceneName + "__0__.sdf";
				fuser.fuse(outCompleteFile, outIncompleteFile, scene, completeFrames, incompleteFrames, debugOut);
			}
			catch (MLibException& e)
			{
				std::stringstream ss;
				ss << "exception caught at scene " << scanInfo.sceneName << " : " << e.what() << std::endl;
				std::cout << ss.str() << std::endl;
			}
			catch (std::exception& e)
			{
				std::stringstream ss;
				ss << "exception caught at scene " << scanInfo.sceneName << " : " << e.what() << std::endl;
				std::cout << ss.str() << std::endl;
			}
		}
	}
	for (unsigned int i = 0; i < scenes.size(); i++) {
		SAFE_DELETE(scenes[i]);
		SAFE_DELETE(fusers[i]);
	}
	std::cout << std::endl << "done!" << std::endl;
}

void Visualizer::render(ApplicationData& app)
{
	m_timer.frame();

	m_font.drawString("FPS: " + convert::toString(m_timer.framesPerSecond()), vec2i(10, 5), 24.0f, RGBColor::Red);

	if (m_bEnableRecording) {
		if (m_recordedCameras.empty()) m_recordedCameras.push_back(std::vector<Cameraf>());
		m_recordedCameras.back().push_back(m_camera);
		m_font.drawString("RECORDING ON " + std::to_string(m_recordedCameras.size()), vec2i(10, 30), 24.0f, RGBColor::Red);
	}
}


void Visualizer::resize(ApplicationData &app)
{
	m_camera.updateAspectRatio((float)app.window.getWidth() / app.window.getHeight());
}

void Visualizer::keyDown(ApplicationData& app, UINT key)
{
	//if (key == KEY_F) app.graphics.castD3D11().toggleWireframe();

	if (key == KEY_U)
	{
	}

	if (key == KEY_I)
	{
		m_scene.randomizeLighting();
	}

	if (key == KEY_Y) {
		m_bEnableAutoRotate = !m_bEnableAutoRotate;
	}

	//record trajectory
	if (key == KEY_R) {
		if (m_bEnableRecording == false) {
			m_recordedCameras.clear();
			m_bEnableRecording = true;
		}
		else {
			m_bEnableRecording = false;
		}
	}

	if (key == KEY_ESCAPE) {
		PostQuitMessage(WM_QUIT);
	}
}

void Visualizer::keyPressed(ApplicationData &app, UINT key)
{
	const float distance = 0.1f;
	const float theta = 0.1f;

	if (key == KEY_S) m_camera.move(-distance);
	if (key == KEY_W) m_camera.move(distance);
	if (key == KEY_A) m_camera.strafe(-distance);
	if (key == KEY_D) m_camera.strafe(distance);
	if (key == KEY_E) m_camera.jump(-distance);
	if (key == KEY_Q) m_camera.jump(distance);

	if (key == KEY_UP) m_camera.lookUp(theta);
	if (key == KEY_DOWN) m_camera.lookUp(-theta);
	if (key == KEY_LEFT) m_camera.lookRight(theta);
	if (key == KEY_RIGHT) m_camera.lookRight(-theta);

	if (key == KEY_Z) m_camera.roll(theta);
	if (key == KEY_X) m_camera.roll(-theta);


}

void Visualizer::mouseDown(ApplicationData &app, MouseButtonType button)
{

}

void Visualizer::mouseWheel(ApplicationData &app, int wheelDelta)
{
	const float distance = 0.01f;
	m_camera.move(distance * wheelDelta);
}

void Visualizer::mouseMove(ApplicationData &app)
{
	const float distance = 0.05f;
	const float theta = 0.5f;

	vec2i posDelta = app.input.mouse.pos - app.input.prevMouse.pos;

	if (app.input.mouse.buttons[MouseButtonRight])
	{
		m_camera.strafe(distance * posDelta.x);
		m_camera.jump(distance * posDelta.y);
	}

	if (app.input.mouse.buttons[MouseButtonLeft])
	{
		m_camera.lookRight(theta * posDelta.x);
		m_camera.lookUp(theta * posDelta.y);
	}

}



