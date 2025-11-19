#pragma once

#include <memory>
#include <boost/container/deque.hpp>

#include <map>
#include <string>

#define boost_deque boost::container::deque

class FrameMetadata;
class Frame;
class Buffer;
class FrameFactory;

typedef std::shared_ptr<FrameMetadata> framemetadata_sp;
typedef std::shared_ptr<Frame> frame_sp;
typedef std::shared_ptr<FrameFactory> framefactory_sp;
typedef std::map<std::string, frame_sp> frame_container;
typedef std::map<std::string, framemetadata_sp> metadata_by_pin;
typedef std::map<std::string, framefactory_sp> framefactory_by_pin;
typedef std::map<std::string, boost_deque<std::string>> Connections;