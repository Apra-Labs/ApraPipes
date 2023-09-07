#pragma once

#include "Module.h"

#include <vtk-9.0/vtkActor.h>
#include <vtk-9.0/vtkCamera.h>
#include <vtk-9.0/vtkCellArray.h>
#include <vtk-9.0/vtkFloatArray.h>
#include <vtk-9.0/vtkNamedColors.h>
#include <vtk-9.0/vtkNew.h>
#include <vtk-9.0/vtkPointData.h>
#include <vtk-9.0/vtkPoints.h>
#include <vtk-9.0/vtkPolyData.h>
#include <vtk-9.0/vtkPolyDataMapper.h>
#include <vtk-9.0/vtkRenderWindow.h>
#include <vtk-9.0/vtkRenderWindowInteractor.h>
#include <vtk-9.0/vtkRenderer.h>
#include <vtk-9.0/vtkTriangleStrip.h>
#include <vtk-9.0/vtkProperty.h>

#include <vtk-9.0/vtkSTLReader.h>

class STLRendererSinkProps : public ModuleProps
{
public:
    STLRendererSinkProps() : ModuleProps()
    {
    }
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