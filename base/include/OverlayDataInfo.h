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
	virtual void serialize(boost::archive::binary_oarchive& oa) {}
	virtual void deserialize(boost::archive::binary_iarchive& ia) { }
	virtual size_t getSerializeSize() { return 0; };
	virtual void accept(OverlayShapeVisitor* visitor) { visitor->visit(this); };

protected:
	Primitive primitiveType;
};

class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE) 
	{
	}

	void serialize(boost::archive::binary_oarchive& oa) override
	{	
		oa << primitiveType << x1 << y1 << radius;
	}

	size_t getSerializeSize()
	{
		return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(primitiveType) + 32;
	}

	void deserialize(boost::archive::binary_iarchive& ia)
	{
		ia >> x1 >> y1 >> radius;
	}

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
	void serialize(boost::archive::binary_oarchive& oa) override
	{
		oa << primitiveType << x1 << y1 << x2 << y2;
	}

	size_t getSerializeSize()
	{
		return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}

	void deserialize(boost::archive::binary_iarchive& ia)
	{
		ia >> x1 >> y1 >> x2 >> y2;
	}

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

	void serialize(boost::archive::binary_oarchive& oa) override
	{
		oa << primitiveType << x1 << y1 << x2 << y2;
	}

	void deserialize(boost::archive::binary_iarchive& ia)
	{
		ia >> x1 >> y1 >> x2 >> y2;
	}

	size_t getSerializeSize()
	{
		return sizeof(RectangleOverlay)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}


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

class OverlayShapeDeserializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeDeserializerVisitor(boost::archive::binary_iarchive& _ia) : ia(_ia) {}
	virtual void visit(OverlayInfo* Overlay, void* buffer, size_t size)
	{
		Overlay->deserialize(ia);
	}

private:
	frame_sp frame;
	boost::archive::binary_iarchive& ia;
};

class OverlayShapeSerializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializerVisitor(boost::archive::binary_oarchive& _oa, frame_sp _frame) : oa(_oa), frame(_frame) {}
	virtual void visit(OverlayInfo* Overlay)
	{
		Overlay->serialize(oa);
	}

protected:
	boost::archive::binary_oarchive& oa;
	frame_sp frame;
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
		OverlayShapeSerializerVisitor* visitor = new OverlayShapeSerializerVisitor(oa,frame);
		oa << gList.size();
		accept(visitor);
	}

	void deserialize(frame_sp frame)
	{
		boost::iostreams::basic_array_source<char> device((char*)frame->data(), frame->size());
		boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
		boost::archive::binary_iarchive ia(s);

		size_t archive_size;
		ia >> archive_size;
		ia >> primitiveType;
		//OverlayInfo* p = factory.create(primitiveType);*/
		RectangleOverlay* recOverlay = new RectangleOverlay();
		recOverlay->deserialize(ia);
		//ia >> recOverlay;
		
	}
protected:
	void accept(OverlayShapeVisitor* visitor)
	{
		for (auto shape : gList)
		{
			shape->accept(visitor);
		}
	}
protected:
	Primitive primitiveType;
	vector<OverlayInfo*> gList;
};