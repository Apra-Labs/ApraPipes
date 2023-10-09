#pragma once

#include "Module.h"

#include <vtk-9.0/vtkActor.h>
#include <vtk-9.0/vtkCamera.h>
#include <vtk-9.0/vtkCellArray.h>
#include <vtk-9.0/vtkNamedColors.h>
#include <vtk-9.0/vtkNew.h>
#include <vtk-9.0/vtkPoints.h>
#include <vtk-9.0/vtkPolyData.h>
#include <vtk-9.0/vtkPolyDataMapper.h>
#include <vtk-9.0/vtkRenderWindow.h>
#include <vtk-9.0/vtkRenderWindowInteractor.h>
#include <vtk-9.0/vtkRenderer.h>
#include <vtk-9.0/vtkProperty.h>

class STLRendererSinkProps : public ModuleProps
{
public:
	STLRendererSinkProps() : ModuleProps()
	{
	}

	~STLRendererSinkProps()
	{}

	// VTK Named Colors: https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html

	// RGB in double: val/255
	double meshDiffuseColor[3] = { 176, 196, 222 }; // LightSteelBlue = {0.69, 0.76, 0.87}

	/* Specular Lighting
	   https://ogldev.org/www/tutorial19/tutorial19.html
	   [[R_spec],             [[R_light],            [[R_Surface],
		[G_spec],     =        [G_light],     *       [G_Surface],   *  M   *   (R.V)^p
		[B_spec]]              [B_light]]             [B_Surface]]
	*/

	// Describes the intensity of specular highlight prop of the object(M). Range: [0-1]
	double meshSpecularCoefficient = 0.3;
	// Describes the shininess factor(p)
	double meshSpecularPower = 60.0;

	int cameraPosition[3] = { 10, 10, 10 };
	int cameraFocalPoint[3] = { 10, 10, 10 };

	std::string winName = "STL_Renderer";

	int winWidth = 600;
	int winHeight = 600;

	// Background color. def: DarkOliveGreen
	double bkgColor[3] = { 85, 107, 47 };
};

class Detail;

class STLRendererSink : public Module
{
public:
	STLRendererSink(STLRendererSinkProps _props);
	virtual ~STLRendererSink();
	bool init();
	bool term();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool setMetadata(framemetadata_sp& inputMetadata);
	bool shouldTriggerSOS();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	boost::shared_ptr<Detail> mDetail;
	STLRendererSinkProps mProps;
};