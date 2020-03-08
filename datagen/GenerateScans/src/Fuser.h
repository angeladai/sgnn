#pragma once

#include "mLibInclude.h"
#include "VoxelGrid.h"

class Fuser
{
public:
	Fuser(ml::ApplicationData& _app);

	~Fuser();
	
	void fuse(const std::string& outputCompleteFile, const std::string& outputIncompleteFile, 
		Scene& scene, const std::vector<unsigned int>& completeFrames,
		const std::vector<unsigned int>& incompleteFrames, bool debugOut = false);
private:

	ml::ApplicationData& m_app;
	D3D11RenderTarget m_renderTarget;
};

