
#pragma once

#include "mLibInclude.h"


struct ScanInfo {
	std::string sceneName;
	std::string meshFile;
	std::string aggregationFile;
	std::string segmentationFile;
	std::vector<std::string> sensFiles;
	std::string alnFile;

	ScanInfo(const std::string& _sceneName, const std::string& _meshFile, const std::string& _aggFile, const std::string& _segFile, const std::vector<std::string>& _sensFiles, const std::string& _alnFile = "") {
		sceneName = _sceneName;
		meshFile = _meshFile;
		aggregationFile = _aggFile;
		segmentationFile = _segFile;
		sensFiles = _sensFiles;
		alnFile = _alnFile;
	}
};

class ScansDirectory {
public:
	ScansDirectory() {}
	~ScansDirectory() {}

	void loadMatterport(std::string scanPath, std::string scanMeshPath, const std::string& sceneListFile, unsigned int maxNumSens) {
		if (scanPath.back() != '/' && scanPath.back() != '\\') scanPath.push_back('/');
		if (scanMeshPath.back() != '/' && scanMeshPath.back() != '\\') scanMeshPath.push_back('/');

		const std::string meshExt = ".reduced.ply";
		const std::string segExt = ".vsegs.json";
		const std::string aggExt = ".semseg.json";
		const std::string sensExt = ".sens";

		const std::string dataSubPath = "region_segmentations";

		clear();
		std::ifstream s(sceneListFile);
		if (!s.is_open()) throw MLIB_EXCEPTION("failed to open " + sceneListFile);
		std::cout << "loading scan info from list..." << std::endl;
		std::string room;
		unsigned int count = 0;
		while (std::getline(s, room)) {
			const auto parts = util::split(room, "_room");
			const std::string& scene = parts[0];
			const std::string& roomId = parts[1];
			std::cout << "\r\t[ " << count++ << "] " << scene << ", " << roomId;
			const std::string meshFile = scanMeshPath + "/" + scene + "/" + dataSubPath + "/region" + roomId + meshExt;
			const std::string aggFile = scanMeshPath + "/" + scene + "/" + dataSubPath + "/region" + roomId + aggExt;
			const std::string segFile = scanMeshPath + "/" + scene + "/" + dataSubPath + "/region" + roomId + segExt;
			
			const std::string sensPath = scanPath + "/" + scene + "/sens";
			std::vector<std::string> sensFiles = { scene + "_0.sens", scene + "_1.sens" , scene + "_2.sens" };
			for (auto& sensFile : sensFiles)
				sensFile = sensPath + "/" + sensFile;
			m_scans.push_back(ScanInfo(room, meshFile, aggFile, segFile, sensFiles));
		}
		std::cout << std::endl;
		std::cout << "[" << util::fileNameFromPath(sceneListFile) << "] | found " << m_scans.size() << " scenes" << std::endl;
	}

	const std::vector<ScanInfo>& getScans() const {
		return m_scans;
	}

	void clear() { m_scans.clear(); }
private:

	std::vector<ScanInfo> m_scans;
};