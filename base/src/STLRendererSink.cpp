#include<boost/serialization/vector.hpp>
#include "STLRendererSink.h"
#include "ApraPoint3f.h"

class Detail
{
private:
	STLRendererSinkProps mProps;
public:
	Detail(STLRendererSinkProps &_props)
	{
		mProps = _props;
	}

	~Detail(){}

	void Detail::renderMesh(vector<ApraPoint3f> &points3D)
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

		// we will render this polyData
		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);
		polyData->SetPolys(triangles);
		
		// vtk pipeline
		vtkNew<vtkPolyDataMapper> meshMapper;
		meshMapper->SetInputData(polyData);
		meshMapper->Update();
		//meshMapper->SetScalarRange(polyData->GetScalarRange());
		vtkNew<vtkActor> meshActor;
		meshActor->SetMapper(meshMapper);
		// meshActor->GetProperty()->SetColor(1.0, 1.0, 1.0); // side effects of setting diffuse and specular color

		vtkNew<vtkNamedColors> colors;
		meshActor->GetProperty()->SetDiffuse(0.8);
		//auto col = colors->GetColor3d("LightSteelBlue").GetData();
		meshActor->GetProperty()->SetDiffuseColor(vtkColor3d(0.69, 0.76, 0.87).GetData()); 
		meshActor->GetProperty()->SetSpecular(0.3);
		meshActor->GetProperty()->SetSpecularPower(60.0);

		// rendering
		vtkNew<vtkCamera> camera;
		camera->SetPosition(10, 10, 10); // todo: prop
		camera->SetFocalPoint(100, 100, 100);

		vtkNew<vtkRenderer> renderer;
		vtkNew<vtkRenderWindow> renWin;
		renWin->AddRenderer(renderer);
		renWin->SetWindowName("mesh"); // todo: prop

		vtkNew<vtkRenderWindowInteractor> iren;
		iren->SetRenderWindow(renWin);

		renderer->AddActor(meshActor);
		renderer->SetBackground(colors->GetColor3d("DarkOliveGreen").GetData()); // todo: prop

		renWin->SetSize(800, 600); // todo: prop

		// interact with data
		renWin->Render();
		iren->Start();
	}

	void Detail::readAndRenderMesh()
	{
		std::string stlfile = "data/RandomMeshTestScene1.stl";

		vtkNew<vtkSTLReader> reader;
		reader->SetFileName(stlfile.c_str());
		reader->Update();

		auto outData = reader->GetOutputDataObject(0);
		auto info = outData->GetInformation();

		// Visualize
		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputConnection(reader->GetOutputPort());

		vtkNew<vtkNamedColors> colors;
		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);
		actor->GetProperty()->SetDiffuse(0.8);
		actor->GetProperty()->SetDiffuseColor(
			colors->GetColor3d("LightSteelBlue").GetData());
		actor->GetProperty()->SetSpecular(0.3);
		actor->GetProperty()->SetSpecularPower(60.0);

		vtkNew<vtkRenderer> renderer;
		vtkNew<vtkRenderWindow> renderWindow;
		renderWindow->AddRenderer(renderer);
		renderWindow->SetWindowName("ReadSTL");

		vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
		renderWindowInteractor->SetRenderWindow(renderWindow);

		renderer->AddActor(actor);
		renderer->SetBackground(colors->GetColor3d("DarkOliveGreen").GetData());

		renderWindow->Render();
		renderWindowInteractor->Start();

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

bool STLRendererSink::process(frame_container &frames)
{
	auto frame = getFrameByType(frames, FrameMetadata::FrameType::POINTS_3D);
	std::vector<ApraPoint3f> points;
	Utils::deSerialize<std::vector<ApraPoint3f>>(points, frame->data(), frame->size());
	mDetail->renderMesh(points);
	//mDetail->readAndRenderMesh();
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

bool STLRendererSink::setMetadata(framemetadata_sp &inputMetadata)
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
