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
	OverlayInfo(Primitive p) : _PrimitiveType(p) {}
	virtual void serialize(void* buffer, size_t size) {}
	virtual void deSerialize(void* buffer, size_t size) { }
	virtual size_t getSerializeSize() { return 0; };
	virtual void accept(OverlayShapeVisitor* visitor) { };

protected:
	Primitive _PrimitiveType;
};

class OverlayShapeVisitor
{
public:
	virtual ~OverlayShapeVisitor() {}
	virtual void visit(OverlayInfo* Overlay) { };
};
 
class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(Primitive::CIRCLE) 
	{
	}

	void serialize(void* buffer, size_t size) override
	{
		auto& info = *this;
		Utils::serialize<CircleOverlay>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(_PrimitiveType) + 32;
	}

	void deSerialize(void* buffer, size_t size)
	{
		CircleOverlay result;

		Utils::deSerialize<CircleOverlay>(result, buffer, size);

		//return &result;
	}

	float x1, y1, radius;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& _PrimitiveType;
		ar &x1 &y1 &radius;
	}


};

class LineOverlay : public OverlayInfo
{
public:
	LineOverlay() : OverlayInfo(Primitive::LINE) 
	{}
	void serialize(void* buffer, size_t size) override
	{
		auto& info = *this;
		Utils::serialize<LineOverlay>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(_PrimitiveType) + 32;
	}

	void deSerialize(void* buffer, size_t size)
	{
		LineOverlay result;

		Utils::deSerialize<LineOverlay>(result, buffer, size);

		//return &result;
	}

	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& _PrimitiveType;
		ar &x1 &y1 &x2 &y2;
	}

};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(Primitive::RECTANGLE) 
	{}

	void serialize(void* buffer, size_t size) override
	{
		auto& info = *this;
		Utils::serialize<RectangleOverlay>(info, buffer, size);
	}

	void deSerialize(void* buffer, size_t size)
	{
		RectangleOverlay result;
		
		Utils::deSerialize<RectangleOverlay>(result, buffer, size);

		//return &result;
	}

	size_t getSerializeSize()
	{
		return sizeof(RectangleOverlay)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(_PrimitiveType) + 32;
	}


	float x1, y1, x2, y2;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& _PrimitiveType;
		ar &x1 &y1 &x2 &y2;
	}

};

class CompositeOverlay : public OverlayInfo
{

public:
	CompositeOverlay() : OverlayInfo(Primitive::COMPOSITE) {}
	void add(OverlayInfo* componentObj)
	{
		gList.push_back(componentObj);
	}

	void accept(OverlayShapeVisitor* visitor)
	{
		for (auto shape : gList)
		{
			visitor->visit(shape);
		}
	}


protected:
	vector<OverlayInfo*> gList;
};

class OverlayShapeDeserializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeDeserializerVisitor(uchar* _buffer, size_t _size) : buffer(_buffer), size(_size) {}
	void visit(OverlayInfo* Overlay)
	{
		Overlay->deSerialize(buffer, size);
	}

private:
	uchar* buffer;
	size_t size;
};

class OverlayShapeSerializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializerVisitor(uchar* _buffer, size_t _size) : buffer(_buffer), size(_size) {}
	void visit(OverlayInfo* Overlay)
	{
		Overlay->serialize((void*)buffer, size);
	}

	
protected:
	uchar* buffer;
	size_t size;
};