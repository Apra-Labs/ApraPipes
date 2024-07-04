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

	STLRendererSinkProps(std::vector<double> _meshDiffuseColor, double _meshSpecularCoefficient, double _meshSpecularPower, std::vector<int> _cameraPosition, std::vector<int> _cameraFocalPoint, std::string _winName ,int _winWidth, int _winHeight, std::vector<double> _bkgColor )
	{
		meshDiffuseColor = _meshDiffuseColor;
		meshSpecularCoefficient = _meshSpecularCoefficient;
		meshSpecularPower = _meshSpecularPower;
		cameraPosition = _cameraPosition;
		cameraFocalPoint = _cameraFocalPoint;
		winName = _winName;
		winWidth = _winWidth;
		winHeight = _winHeight;
		bkgColor = _bkgColor;

	}
	std::vector<double> meshDiffuseColor;
	double meshSpecularCoefficient;
	double meshSpecularPower;
	std::vector<int> cameraPosition;
	std::vector<int> cameraFocalPoint;
	std::string winName;
	int winWidth;
	int winHeight;
	std::vector<double> bkgColor;

	size_t getSerializeSize()
	{
		return ModuleProps::getSerializeSize() + sizeof(int) * 2 + sizeof(double) *2 + sizeof(winName) + sizeof(meshDiffuseColor)+ sizeof(cameraPosition)+ sizeof(cameraFocalPoint)+ sizeof(bkgColor);
	}


private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar &boost::serialization::base_object<ModuleProps>(*this);
		ar &meshDiffuseColor;
		ar &meshSpecularCoefficient; 
		ar &meshSpecularPower;
		ar &cameraPosition ;
		ar &cameraFocalPoint;
		ar &winName;
		ar &winWidth ;
		ar &winHeight;
		ar &bkgColor;
	}
};

class STLRendererSink : public Module
{
public:
	STLRendererSink(STLRendererSinkProps _props);
	virtual ~STLRendererSink();
	bool init();
	bool term();
	void setProps(STLRendererSinkProps &props);
	STLRendererSinkProps getProps();
protected:
	bool process(frame_container& frames);
	bool processSOS(frame_sp& frame);
	bool validateInputPins();
	bool setMetadata(framemetadata_sp& inputMetadata);
	bool shouldTriggerSOS();
	void addInputPin(framemetadata_sp& metadata, string& pinId);
	bool handlePropsChange(frame_sp& frame);

private:
	class Detail;
	boost::shared_ptr<Detail> mDetail;
};