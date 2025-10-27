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
	DRAWING,
	TEXT
};

class OverlayInfo;

class OverlayInfoVisitor
{
public:
	virtual ~OverlayInfoVisitor() {}
	virtual void visit(OverlayInfo* overlay) {};
};

class OverlayInfo
{
public:
	OverlayInfo(Primitive p) : primitiveType(p) {}
	OverlayInfo() {}
	virtual void serialize(boost::archive::binary_oarchive& oa) {}
	virtual void deserialize(boost::archive::binary_iarchive& ia) {}
	virtual size_t getSerializeSize() { return 0; }
	virtual void accept(OverlayInfoVisitor* visitor) { visitor->visit(this); };
	virtual void draw(cv::Mat matImg) {}
	Primitive primitiveType;
};

class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE) {}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);
	void draw(cv::Mat matImg);

	float x1, y1, radius;
private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& radius;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x1& y1& radius;
	}
};

class LineOverlay : public OverlayInfo
{
public:
	LineOverlay() : OverlayInfo(Primitive::LINE) {}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);
	void draw(cv::Mat matImg);

	float x1, y1, x2, y2;
private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(Primitive::RECTANGLE) {}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);
	void draw(cv::Mat matImg);

	float x1, y1, x2, y2;
private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
};

class TextOverlay : public OverlayInfo
{
public:
	TextOverlay() : OverlayInfo(Primitive::TEXT), fontSize(0.5) {}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);
	void draw(cv::Mat matImg);

	float x, y;
	std::string text;
	float fontSize;
private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar& primitiveType;
		ar& x& y& text& fontSize;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x& y& text& fontSize;
	}
};

// visitorsheirarchy
class OverlayInfoSerializerVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoSerializerVisitor(boost::archive::binary_oarchive& _oa) : oa(_oa) {}
	void visit(OverlayInfo* overlay);
private:
	boost::archive::binary_oarchive& oa;
};

// visitor to estimate serialize size
class OverlayInfoSerializeSizeVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoSerializeSizeVisitor() : totalSize(0) {}
	void visit(OverlayInfo* overlay);
	size_t totalSize;
};

class OverlayInfoDrawingVisitor : public OverlayInfoVisitor
{
public:
	OverlayInfoDrawingVisitor(frame_sp frame)
	{
		matImg = Utils::getMatHeader(FrameMetadataFactory::downcast<RawImageMetadata>(frame->getMetadata()));
		matImg.data = static_cast<uchar*>(frame->data());
	}
	void visit(OverlayInfo* overlay);
private:
	cv::Mat matImg;
};

class CompositeOverlay : public OverlayInfo
{
public:
	CompositeOverlay() : OverlayInfo(Primitive::COMPOSITE) {}
	CompositeOverlay(Primitive primitiveType) : OverlayInfo(primitiveType) {}
	void add(OverlayInfo* component);
	// used by builder
	void deserialize(boost::archive::binary_iarchive& ia);
	void accept(OverlayInfoVisitor* visitor);
	vector<OverlayInfo*> getList();

protected:
	vector<OverlayInfo*> gList; // used by DrawingOverlay
	// used by visitor
	void serialize(boost::archive::binary_oarchive& oa);

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar& primitiveType;
		ar& gList.size();
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version /*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& gList.size();
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
	void draw(frame_sp frame);
	size_t mGetSerializeSize();
};

// Builder heirarchy
class DrawingOverlayBuilder
{
public:
	DrawingOverlayBuilder() {}
	virtual OverlayInfo* deserialize(boost::archive::binary_iarchive& ia) = 0;
};

class CompositeOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CompositeOverlayBuilder() : compositeOverlay(new CompositeOverlay()) {}
	OverlayInfo* deserialize(boost::archive::binary_iarchive& ia);

protected:
	CompositeOverlay* compositeOverlay;
};

class LineOverlayBuilder : public DrawingOverlayBuilder
{
public:
	LineOverlayBuilder() : lineOverlay(new LineOverlay()) {}
	OverlayInfo* deserialize(boost::archive::binary_iarchive& ia);

protected:
	LineOverlay* lineOverlay;
};

class RectangleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	RectangleOverlayBuilder() : rectangleOverlay(new RectangleOverlay()) {}
	OverlayInfo* deserialize(boost::archive::binary_iarchive& ia);

protected:
	RectangleOverlay* rectangleOverlay;
};

class CircleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CircleOverlayBuilder() : circleOverlay(new CircleOverlay()) {}
	OverlayInfo* deserialize(boost::archive::binary_iarchive& ia);

protected:
	CircleOverlay* circleOverlay;
};

class TextOverlayBuilder : public DrawingOverlayBuilder
{
public:
	TextOverlayBuilder() : textOverlay(new TextOverlay()) {}
	OverlayInfo* deserialize(boost::archive::binary_iarchive& ia);

protected:
	TextOverlay* textOverlay;
};
