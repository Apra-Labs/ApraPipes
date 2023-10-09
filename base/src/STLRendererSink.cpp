#include<boost/serialization/vector.hpp>
#include "STLRendererSink.h"
#include "ApraPoint3f.h"

class Detail
{
private:
	STLRendererSinkProps mProps;
public:
	Detail(STLRendererSinkProps& _props)
	{
		mProps = _props;
	}

	~Detail()
	{
		LOG_TRACE << "STLRenderer::Detail destructor";
	}

	void renderMesh(vector<ApraPoint3f>& points3D)
	{
		// https://www.kitware.com/a-tour-of-vtk-pointer-classes/

		auto nPoints = points3D.size();
		vtkNew<vtkPoints> points;
		for (auto i = 0; i < nPoints; ++i)
		{
			auto& pts = points3D[i];
			points->InsertNextPoint(pts.x, pts.y, pts.z);
		}

		// define topography using point IDs (indices)
		vtkNew<vtkCellArray> triangles;
		for (vtkIdType i = 0; i < nPoints; i += 3)
		{
			vtkIdType ids[3] = { i, i + 1, i + 2 };
			triangles->InsertNextCell(3, ids);
		}

		LOG_INFO << "nPoints <" << points->GetNumberOfPoints() << ">";
		LOG_INFO << "nTriangles <" << triangles->GetNumberOfCells() << ">";

		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);
		polyData->SetPolys(triangles);

		// vtk pipeline
		vtkNew<vtkPolyDataMapper> meshMapper;
		meshMapper->SetInputData(polyData);
		meshMapper->Update();

		vtkNew<vtkActor> meshActor;
		meshActor->SetMapper(meshMapper);

		vtkNew<vtkNamedColors> colors;
		meshActor->GetProperty()->SetDiffuse(0.8);
		//auto col = colors->GetColor3d("LightSteelBlue").GetData();
		double meshDiffuseColor[3] = { mProps.meshDiffuseColor[0] / 255.0,
			mProps.meshDiffuseColor[1] / 255.0,
			mProps.meshDiffuseColor[2] / 255.0 };
		meshActor->GetProperty()->SetDiffuseColor(vtkColor3d(meshDiffuseColor[0], meshDiffuseColor[1], meshDiffuseColor[2]).GetData());
		meshActor->GetProperty()->SetSpecular(mProps.meshSpecularCoefficient);
		meshActor->GetProperty()->SetSpecularPower(mProps.meshSpecularPower);

		// rendering
		vtkNew<vtkCamera> camera;
		camera->SetPosition(mProps.cameraPosition[0], mProps.cameraPosition[1], mProps.cameraPosition[2]);
		camera->SetFocalPoint(mProps.cameraFocalPoint[0], mProps.cameraFocalPoint[1], mProps.cameraPosition[2]);

		vtkNew<vtkRenderer> renderer;
		vtkNew<vtkRenderWindow> renWin;
		renWin->AddRenderer(renderer);
		renWin->SetWindowName(mProps.winName.c_str());

		vtkNew<vtkRenderWindowInteractor> iren;
		iren->SetRenderWindow(renWin);

		renderer->AddActor(meshActor);
		double bkgColor[3] = { mProps.bkgColor[0] / 255.0,
			mProps.bkgColor[1] / 255.0,
			mProps.bkgColor[2] / 255.0 };
		renderer->SetBackground(vtkColor3d(bkgColor[0], bkgColor[1], bkgColor[2]).GetData());

		renWin->SetSize(mProps.winWidth, mProps.winHeight);

		// interact with data
		renWin->Render();
		iren->Start();
		iren->GetRenderWindow()->Finalize();
		iren->TerminateApp();
	}

};

STLRendererSink::STLRendererSink(STLRendererSinkProps props) :
	Module(SINK, "STLRendererSink", props), mProps(props)
{
	mDetail.reset(new Detail(props));
}

STLRendererSink::~STLRendererSink()
{
}

bool STLRendererSink::init()
{
	if (!Module::init())
	{
		return false;
	}
	return true;
}

bool STLRendererSink::term()
{
	return true;
}

bool STLRendererSink::process(frame_container& frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::FrameType::POINTS_3D);
	std::vector<ApraPoint3f> points;
	Utils::deSerialize<std::vector<ApraPoint3f>>(points, frame->data(), frame->size());
	mDetail->renderMesh(points);
	return true;
}

bool STLRendererSink::processSOS(frame_sp& frame)
{
	return true;
}

bool STLRendererSink::validateInputPins()
{
	if (getNumberOfInputPins() != 1)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins size is expected to be 1. Actual<" << getNumberOfInputPins() << ">";
		return false;
	}

	framemetadata_sp metadata = getFirstInputMetadata();
	FrameMetadata::FrameType frameType = metadata->getFrameType();
	if (frameType != FrameMetadata::FrameType::POINTS_3D)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input frameType is expected to be RAW_IMAGE. Actual<" << frameType << ">";
		return false;
	}

	FrameMetadata::MemType memType = metadata->getMemType();
	if (memType != FrameMetadata::MemType::HOST)
	{
		LOG_ERROR << "<" << getId() << ">::validateInputPins input memType is expected to be HOST. Actual<" << memType << ">";
		return false;
	}
	return true;
}

bool STLRendererSink::setMetadata(framemetadata_sp& inputMetadata)
{
	return true;
}

bool STLRendererSink::shouldTriggerSOS()
{
	return false;
}

void STLRendererSink::addInputPin(framemetadata_sp& metadata, string& pinId)
{
	Module::addInputPin(metadata, pinId);
}
