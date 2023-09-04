#include "STLRendererSink.h"

class Detail
{
public:
	Detail(STLRendererSinkProps _props)
	{

	}

	~Detail(){}

	void Detail::render()
	{
		vtkNew<vtkNamedColors> colors;

		std::array<std::array<double, 3>, 8> pts = { {{{0, 0, 0}},
													 {{1, 0, 0}},
													 {{1, 1, 0}},
													 {{0, 1, 0}},
													 {{0, 0, 1}},
													 {{1, 0, 1}},
													 {{1, 1, 1}},
													 {{0, 1, 1}}} };
		// The ordering of the corner points on each face.
		std::array<std::array<vtkIdType, 4>, 6> ordering = { {{{0, 3, 2, 1}},
															 {{4, 5, 6, 7}},
															 {{0, 1, 5, 4}},
															 {{1, 2, 6, 5}},
															 {{2, 3, 7, 6}},
															 {{3, 0, 4, 7}}} };

		// We'll create the building blocks of polydata including data attributes.
		vtkNew<vtkPolyData> cube;
		vtkNew<vtkPoints> points;
		vtkNew<vtkCellArray> polys;
		vtkNew<vtkFloatArray> scalars;

		// Load the point, cell, and data attributes.
		for (auto i = 0ul; i < pts.size(); ++i)
		{
			points->InsertPoint(i, pts[i].data());
			scalars->InsertTuple1(i, i);
		}
		for (auto&& i : ordering)
		{
			polys->InsertNextCell(vtkIdType(i.size()), i.data());
		}

		// We now assign the pieces to the vtkPolyData.
		cube->SetPoints(points);
		cube->SetPolys(polys);
		cube->GetPointData()->SetScalars(scalars);

		// Now we'll look at it.
		vtkNew<vtkPolyDataMapper> cubeMapper;
		cubeMapper->SetInputData(cube);
		cubeMapper->SetScalarRange(cube->GetScalarRange());
		vtkNew<vtkActor> cubeActor;
		cubeActor->SetMapper(cubeMapper);

		// The usual rendering stuff.
		vtkNew<vtkCamera> camera;
		camera->SetPosition(1, 1, 1);
		camera->SetFocalPoint(0, 0, 0);

		vtkNew<vtkRenderer> renderer;
		vtkNew<vtkRenderWindow> renWin;
		renWin->AddRenderer(renderer);
		renWin->SetWindowName("Cube");

		vtkNew<vtkRenderWindowInteractor> iren;
		iren->SetRenderWindow(renWin);

		renderer->AddActor(cubeActor);
		renderer->SetActiveCamera(camera);
		renderer->ResetCamera();
		renderer->SetBackground(colors->GetColor3d("Cornsilk").GetData());

		renWin->SetSize(600, 600);

		// interact with data
		renWin->Render();
		iren->Start();

		return;
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
	mDetail->render();
	return true;
}

bool STLRendererSink::processSOS(frame_sp& frame)
{
	return true;
}


bool STLRendererSink::validateInputPins()
{
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
	return;
}
