#pragma once

#include <boost/serialization/vector.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include "Utils.h"
#include "Module.h"


enum Primitive
{
	RECTANGLE,
	CIRCLE,
	LINE,
	COMPOSITE,
	DRAWING
};

class OverlayInfo;

class OverlayInfoVisitor
{
public:
	virtual ~OverlayInfoVisitor() {}
	virtual void visit(OverlayInfo *Overlay){};
};

class OverlayInfo
{
public:
	OverlayInfo(Primitive p) : primitiveType(p) {}
	OverlayInfo() {}
	virtual void serialize(boost::archive::binary_oarchive &oa) {}
	virtual void deserialize(boost::archive::binary_iarchive &ia) {}
	virtual size_t getSerializeSize() { return 0; }
	virtual void accept(OverlayInfoVisitor *visitor) { visitor->visit(this); };
	virtual void draw(frame_sp frame)
	{
	  iImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(frame->getMetadata()));
	};
	Primitive primitiveType;
	cv::Mat iImg;
};

class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE)
	{
	}

	void serialize(boost::archive::binary_oarchive &oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive &ia);

	void draw(frame_sp frame) override
	{
		OverlayInfo::draw(frame);
		cv::Point p(x1, y1);
		circle(iImg, p, radius, cv::Scalar(255, 200, 0), 2);
	};

	float x1, y1, radius;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive &ar, const unsigned int version /*file_version*/)
	{
		ar &primitiveType;
		ar &x1 &y1 &radius;
	}
	template <class Archive>
	void load(Archive &ar, const unsigned int version)
	{
		if (version > 0)
			ar &primitiveType;
		ar &x1 &y1 &radius;
	}
};

class LineOverlay : public OverlayInfo
{
public:
	LineOverlay() : OverlayInfo(Primitive::LINE)
	{
	}
	void serialize(boost::archive::binary_oarchive &oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive &ia);
	void draw(frame_sp frame) override
	{
		OverlayInfo::draw(frame);
		cv::Point point1(x1, y1);
		cv::Point point2(x2, y2);
		line(iImg, point1, point2, cv::Scalar(255, 0, 0), 2);
	};
	
	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive &ar, const unsigned int version /*file_version*/)
	{
		ar &primitiveType;
		ar &x1 &y1 &x2 &y2;
	}
	template <class Archive>
	void load(Archive &ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar &primitiveType;
		ar &x1 &y1 &x2 &y2;
	}
};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(Primitive::RECTANGLE)
	{
	}
	void serialize(boost::archive::binary_oarchive &oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive &ia);

	void draw(frame_sp frame) override
	{
		OverlayInfo::draw(frame);
		cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
		cv::rectangle(iImg, rect, cv::Scalar(0, 255, 0), 2);
	};
	
	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive &ar, const unsigned int version /*file_version*/)
	{
		ar &primitiveType;
		ar &x1 &y1 &x2 &y2;
	}
	template <class Archive>
	void load(Archive &ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar &primitiveType;
		ar &x1 &y1 &x2 &y2;
	}
};

// visitors heirarchy
class OverlayInfoSerializerVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoSerializerVisitor(boost::archive::binary_oarchive &_oa) : oa(_oa)
	{
	}
	void visit(OverlayInfo *overlay) override
	{
		overlay->serialize(oa);
	}

private:
	boost::archive::binary_oarchive &oa;
};

// visitor to estimate serialize size
class OverlayInfoSerializeSizeVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoSerializeSizeVisitor() : totalSize(0) {}

	void visit(OverlayInfo *overlay) override
	{
		totalSize += overlay->getSerializeSize();
	}

	size_t totalSize;
};

class OverlayInfoDrawingVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoDrawingVisitor(frame_sp frame) : mframe(frame) {}


	void visit(OverlayInfo* overlay) override
	{
		overlay->draw(mframe);
	}

	frame_sp mframe;
};

class CompositeOverlay : public OverlayInfo
{
public:
	CompositeOverlay() : OverlayInfo(Primitive::COMPOSITE) {}
	CompositeOverlay(Primitive primitiveType) : OverlayInfo(primitiveType) {}
	void add(OverlayInfo *component);
	// used by builder
	void deserialize(boost::archive::binary_iarchive &ia);
	void accept(OverlayInfoVisitor *visitor);
	vector<OverlayInfo *> getList();

protected:
	vector<OverlayInfo *> gList; // used by DrawingOverlay
	// used by visitor
	void serialize(boost::archive::binary_oarchive &oa);

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive &ar, const unsigned int version /*file_version*/)
	{
		ar &primitiveType;
		ar &gList.size();
	}
	template <class Archive>
	void load(Archive &ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar &primitiveType;
		ar &gList.size();
	}
};

// interface to be used externally
// why is it required ? - put commment here
class DrawingOverlay : public CompositeOverlay
{
public:
	DrawingOverlay() : CompositeOverlay(Primitive::DRAWING) {}
	void serialize(frame_sp frame);
	void deserialize(frame_sp frame);
	void mdraw(frame_sp frame) ;
	size_t mGetSerializeSize();
};

// Builder heirarchy
class DrawingOverlayBuilder
{
public:
	DrawingOverlayBuilder() {}
	virtual OverlayInfo *deserialize(boost::archive::binary_iarchive &ia) = 0;
};

class CompositeOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CompositeOverlayBuilder() : compositeOverlay(new CompositeOverlay()) {}
	OverlayInfo *deserialize(boost::archive::binary_iarchive &ia);

protected:
	CompositeOverlay *compositeOverlay;
};

class LineOverlayBuilder : public DrawingOverlayBuilder
{
public:
	LineOverlayBuilder() : lineOverlay(new LineOverlay()) {}
	OverlayInfo *deserialize(boost::archive::binary_iarchive &ia);

protected:
	LineOverlay *lineOverlay;
};

class RectangleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	RectangleOverlayBuilder() : rectangleOverlay(new RectangleOverlay()) {}
	OverlayInfo *deserialize(boost::archive::binary_iarchive &ia);

protected:
	RectangleOverlay *rectangleOverlay;
};

class CircleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CircleOverlayBuilder() : circleOverlay(new CircleOverlay()) {}
	OverlayInfo *deserialize(boost::archive::binary_iarchive &ia);

protected:
	CircleOverlay *circleOverlay;
};
