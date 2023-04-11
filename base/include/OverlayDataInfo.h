#pragma once

#include <boost/serialization/vector.hpp>
#include "Utils.h"
#include "Module.h"

enum primitive
{
	COMPOSITE = 0,
	CIRCLE,
	LINE,
	RECTANGLE
};

class RectangleOverlay;
class CircleOverlay;
class LineOverlay;

class OverlayShapeVisitor
{
public:
	virtual ~OverlayShapeVisitor() {}
	virtual void visit(OverlayInfo* Overlay) = 0;
};

class OverlayInfo
{
public:
	OverlayInfo(primitive p) : _primitiveType(p) {}
	virtual void serialize(void* buffer, size_t size) {}
	virtual void deSerialize(void* buffer, size_t size) {}
	virtual size_t getSerializeSize() { return 0; };
	virtual void accept(OverlayShapeVisitor* visitor) {};

protected:
	primitive _primitiveType;

};
 
class CircleOverlay : public OverlayInfo
{
public:
	CircleOverlay() : OverlayInfo(primitive::CIRCLE) { }

	void serialize(void* buffer, size_t size)
	{
		auto& info = *this;
		Utils::serialize<CircleOverlay>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		return sizeof(CircleOverlay) + sizeof(x1) + sizeof(y1) + sizeof(radius) + sizeof(_primitiveType) + 32;
	}

	float x1, y1, radius;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& x1& y1& x2& y2;
	}


};

class LineOverlay : public OverlayInfo
{
public:
	LineOverlay() : OverlayInfo(primitive::LINE) { }
	void serialize(void* buffer, size_t size)
	{
		auto& info = *this;
		Utils::serialize<LineOverlay>(info, buffer, size);
	}

	size_t getSerializeSize()
	{
		return sizeof(LineOverlay) + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}

	float x1, y1, x2, y2;
	primitive primitiveType;
	std::vector<LineOverlay> vectorRect;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& x2& y2;
	}

};

class RectangleOverlay : public OverlayInfo
{
public:
	RectangleOverlay() : OverlayInfo(primitive::RECTANGLE) { }
	RectangleOverlay(primitive _primitiveType) {
		primitiveType = _primitiveType;
	}

	void serialize(void* buffer, size_t size)
	{
		auto& info = *this;
		Utils::serialize<RectangleOverlay>(info, buffer, size);
	}

	RectangleOverlay deSerialize(void* buffer, size_t size)
	{
		RectangleOverlay result;
		
		Utils::deSerialize<RectangleOverlay>(result, buffer, size);

		return result;
	}

	size_t getSerializeSize()
	{
		return sizeof(RectangleOverlay)  + sizeof(x1) + sizeof(y1) + sizeof(x2) + sizeof(y2) + sizeof(primitiveType) + 32;
	}


	float x1, y1, x2, y2;
	primitive primitiveType;
	std::vector<RectangleOverlay> vectorRect;

private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*file_version*/)
	{
		ar& primitiveType;
		ar& x1& y1& x2& y2;
	}

};

class CompositeOverlay : public OverlayInfo
{

public:
	void add(OverlayInfo componentObj)
	{
		gList.push_back(componentObj);
	}

	virtual void accept(OverlayShapeVisitor* visitor)
	{
		for (auto shape : gList)
		{
			visitor->visit(&shape);
		}
	}

private:
	vector<OverlayInfo> gList;
};

class OverlayShapeDeserializerVisitor : public OverlayShapeVisitor
{
public:
	void visit(OverlayInfo Overlay)
	{
		
	}

	OverlayInfo get()
	{
		return this->info;
	}

private:
	OverlayInfo info;
};

class OverlayShapeSerializerVisitor : public OverlayShapeVisitor
{
public:
	OverlayShapeSerializerVisitor(uchar* _buffer, size_t _size) : buffer(_buffer), size(_size) {}
	void visit(OverlayInfo Overlay)
	{
		Overlay.serialize(buffer, size);
	}

	
protected:
	uchar* buffer;
	size_t size;
};