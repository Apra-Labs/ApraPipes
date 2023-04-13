#pragma once

#include <boost/serialization/vector.hpp>
#include "Utils.h"
#include "Module.h"

enum Primitive
{
	COMPOSITE = 0,
	CIRCLE,
	LINE,
	RECTANGLE
};

class OverlayShapeVisitor;

class OverlayInfo
{
public:
	OverlayInfo(Primitive p) : primitiveType(p) {}
	virtual void serialize(boost::archive::binary_oarchive& oa, void* buffer, size_t size) {}
	virtual void deserialize(boost::archive::binary_iarchive& ia, void* buffer, size_t size) { }
	virtual size_t getSerializeSize() { return 0; };
	virtual void accept(OverlayShapeVisitor* visitor) { };

protected:
	Primitive primitiveType;
};

class OverlayShapeVisitor
{
public:
	virtual ~OverlayShapeVisitor() {}
	virtual void visit(OverlayInfo* Overlay, void* buffer, size_t size) { };
};
 
class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE) 
	{
	}

	void serialize(boost::archive::binary_oarchive& oa, void* buffer, size_t size) override
	{
		
		oa << primitiveType << x1 << y1 << radius;
	}

	size_t getSerializeSize()
	{
		return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(primitiveType) + 32;
	}

	void deserialize(boost::archive::binary_iarchive& ia, void* buffer, size_t size)
	{
		ia >> primitiveType >> x1 >> y1 >> radius;
	}

	float x1, y1, radius;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar & primitiveType;
		ar &x1 &y1 &radius;
	}


};

class LineOverlay : public OverlayInfo
{
public:
	LineOverlay() : OverlayInfo(Primitive::LINE) 
	{}
	void serialize(boost::archive::binary_oarchive& oa, void* buffer, size_t size) override
	{
		oa << primitiveType << x1 << y1 << x2 << y2;
	}

	size_t getSerializeSize()
	{
		return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}

	void deserialize(boost::archive::binary_iarchive& ia, void* buffer, size_t size)
	{
		ia >> primitiveType >> x1 >> y1 >> x2 >> y2;
	}

	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& primitiveType;
		ar &x1 &y1 &x2 &y2;
	}

};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(Primitive::RECTANGLE) 
	{}

	void serialize(boost::archive::binary_oarchive& oa, void* buffer, size_t size) override
	{
		oa << primitiveType << x1 << y1 << x2 << y2;
	}

	void deserialize(boost::archive::binary_iarchive& ia, void* buffer, size_t size)
	{
		ia >> primitiveType >> x1 >> y1 >> x2 >> y2;
	}

	size_t getSerializeSize()
	{
		return sizeof(RectangleOverlay)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}


	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& primitiveType;
		ar &x1 &y1 &x2 &y2;
	}

};

class OverlayShapeDeserializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeDeserializerVisitor(boost::archive::binary_iarchive& _ia) : ia(_ia) {}
	void visit(OverlayInfo* Overlay, void* buffer, size_t size)
	{
		Overlay->deserialize(ia, buffer, size);
	}

private:
	boost::archive::binary_iarchive& ia;
};

class OverlayShapeSerializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializerVisitor(boost::archive::binary_oarchive& _oa) : oa(_oa) {}
	void visit( OverlayInfo* Overlay, void* buffer, size_t size)
	{
		Overlay->serialize(oa, buffer, size);
	}

protected:
	boost::archive::binary_oarchive& oa;
};

class CompositeOverlay : public OverlayInfo
{
public:
	CompositeOverlay() : OverlayInfo(Primitive::COMPOSITE) {}
	void add(OverlayInfo* componentObj)
	{
		gList.push_back(componentObj);
	}

	void serialize(frame_sp frame)
	{
		boost::iostreams::basic_array_sink<char> device_sink((char*)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_sink<char> > s_sink(device_sink);

		boost::archive::binary_oarchive oa(s_sink);
		OverlayShapeSerializerVisitor* seriliazerobj = new OverlayShapeSerializerVisitor(oa);
		accept(seriliazerobj, frame->data(), frame->size());
	}

	void deserialize(frame_sp frame)
	{
		boost::iostreams::basic_array_source<char> device((char*)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
		boost::archive::binary_iarchive ia(s);

		OverlayShapeDeserializerVisitor* deseriliazerobj = new OverlayShapeDeserializerVisitor(ia);
		accept(deseriliazerobj, frame->data(), frame->size());
	}
protected:
	void accept(OverlayShapeVisitor* visitor, void* buffer, size_t size)
	{
		for (auto shape : gList)
		{
			visitor->visit(shape, buffer, size);
		}
	}
protected:
	vector<OverlayInfo*> gList;
};