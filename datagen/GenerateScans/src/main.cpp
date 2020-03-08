

#include "stdafx.h"


void loadGlobalAppState(const std::string& fileNameDescGlobalApp) {
	if (!util::fileExists(fileNameDescGlobalApp)) {
		throw MLIB_EXCEPTION("cannot find parameter file " + fileNameDescGlobalApp);
	}

	std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
	ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
	GlobalAppState::get().readMembers(parameterFileGlobalApp);
	GlobalAppState::get().print();
}

int _tmain(int argc, _TCHAR* argv[])
{
	try {
		std::string fileNameDescGlobalApp;
		if (argc == 2) {
			auto ws = std::wstring(argv[1]);
			fileNameDescGlobalApp = std::string(ws.begin(), ws.end());
		}
		else {
			fileNameDescGlobalApp = "zParametersScanMP.txt";
		}
		loadGlobalAppState(fileNameDescGlobalApp);

		Visualizer callback;
		ApplicationWin32 app(NULL, 640, 480, "Virtual Scan", GraphicsDeviceTypeD3D11, callback);
		app.messageLoop();
	}
	catch (const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	return 0;
}
