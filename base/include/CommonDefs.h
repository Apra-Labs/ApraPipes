#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>

#include <map>

#define boost_deque boost::container::deque

class FrameMetadata;
class Frame;
class Buffer;

typedef boost::shared_ptr<FrameMetadata> framemetadata_sp;
typedef boost::shared_ptr<Frame> frame_sp;
typedef boost::shared_ptr<Buffer> buffer_sp;
typedef std::map<std::string, frame_sp> frame_container;
typedef std::map<std::string, framemetadata_sp> metadata_by_pin;
typedef std::map<std::string, boost_deque<std::string>> Connections;