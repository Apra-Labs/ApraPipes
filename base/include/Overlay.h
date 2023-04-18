#pragma once

#include <boost/serialization/vector.hpp>
#include "Utils.h"
#include "Module.h"

enum Primitive
{
	RECTANGLE,
	CIRCLE,
	LINE,
	COMPOSITE
};

class OverlayInfo;

class OverlayShapeVisitor
{
public:
	virtual ~OverlayShapeVisitor() {}
	virtual void visit(OverlayInfo* Overlay) { };
};
 
class OverlayInfo
{
public:
	OverlayInfo(Primitive p) : primitiveType(p) {}
	OverlayInfo() {}
	virtual void serialize(boost::archive::binary_oarchive& oa) {}
	virtual void deserialize(boost::archive::binary_iarchive& ia) { }
	virtual size_t getSerializeSize() { return 0; };
	virtual void accept(OverlayShapeVisitor* visitor) { visitor->visit(this); };

	Primitive primitiveType;
};

class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE) 
	{}

	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);

	float x1, y1, radius;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version /*file_version*/)
	{
		ar & primitiveType;
		ar &x1 &y1 &radius;
	}
	template<class Archive>
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
	LineOverlay() : OverlayInfo(Primitive::LINE) 
	{}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);

	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version/*file_version*/)
	{
		ar& primitiveType;
		ar &x1 &y1 &x2 &y2;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version/*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(Primitive::RECTANGLE)
	{}
	void serialize(boost::archive::binary_oarchive& oa);
	size_t getSerializeSize();
	void deserialize(boost::archive::binary_iarchive& ia);

	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void save(Archive& ar, const unsigned int version/*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
	template <class Archive>
	void load(Archive& ar, const unsigned int version/*file_version*/)
	{
		if (version > 0)
			ar& primitiveType;
		ar& x1& y1& x2& y2;
	}
};

class OverlayShapeSerializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializerVisitor(boost::archive::binary_oarchive& _oa) : oa(_oa)
	{ }
	virtual void visit(OverlayInfo* overlay)
	{
		overlay->serialize(oa);
	}
private:
	boost::archive::binary_oarchive& oa;
};

class OverlayShapeSerializeSizeVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializeSizeVisitor() : totalSize(0) {}

	virtual void visit(OverlayInfo* Overlay)
	{
		totalSize += Overlay->getSerializeSize();
	}

	size_t totalSize;

};

class CompositeOverlay : public OverlayInfo
{
public:
	CompositeOverlay() : OverlayInfo(Primitive::COMPOSITE) {}
	void add(OverlayInfo* componentObj);
	void serialize(frame_sp frame);
	virtual void deserialize(frame_sp frame) {}
	void accept(OverlayShapeVisitor* visitor);
	vector<OverlayInfo*> gList;
	friend class DrawingOverlay;
	void deserialize(boost::archive::binary_iarchive& ia);
	size_t getSerializeSize()
	{
		OverlayShapeSerializeSizeVisitor* visitor = new OverlayShapeSerializeSizeVisitor();
		accept(visitor);
		return visitor->totalSize;
	}
};

class DrawingOverlayBuilder;

class DrawingOverlay : public CompositeOverlay
{
public:
	DrawingOverlay() {}
	void add(OverlayInfo* componentObj);

	void deserialize(frame_sp frame) override;

};

class DrawingOverlayBuilder
{
public:
	DrawingOverlayBuilder() : m_drawingOverlay(new DrawingOverlay()) {}

	virtual void deserialize(boost::archive::binary_iarchive& ia) = 0;

protected:
	DrawingOverlay* m_drawingOverlay;
};

class CompositeOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CompositeOverlayBuilder() : compositeOverlay(new CompositeOverlay()) {}
	void deserialize(boost::archive::binary_iarchive& ia) override
	{
		compositeOverlay->deserialize(ia);
	}
protected:
	CompositeOverlay* compositeOverlay;
};

class LineOverlayBuilder : public DrawingOverlayBuilder
{
public:
	LineOverlayBuilder() : lineOverlay(new LineOverlay()) {}
	void deserialize(boost::archive::binary_iarchive& ia) override
	{
		lineOverlay->deserialize(ia);
	}
protected:
	LineOverlay* lineOverlay;
};

class RectangleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	RectangleOverlayBuilder() : rectangleOverlay(new RectangleOverlay()) {}
	void deserialize(boost::archive::binary_iarchive& ia) override
	{
		rectangleOverlay->deserialize(ia);
	}
protected:
	RectangleOverlay* rectangleOverlay;
};

class CircleOverlayBuilder : public DrawingOverlayBuilder
{
public:
	CircleOverlayBuilder() : circleOverlay(new CircleOverlay()) {}
	void deserialize(boost::archive::binary_iarchive& ia) override
	{
		circleOverlay->deserialize(ia);
	}
protected:
	CircleOverlay* circleOverlay;
};

class BuilderOverlayFactory
{
public:
	static DrawingOverlayBuilder* create(Primitive primitiveType);
};