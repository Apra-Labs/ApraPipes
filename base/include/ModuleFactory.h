#pragma once
#include "TypeFactory.h"
class Module;
class ModuleFactory
{
	TypeFactory<Module*, size_t, boost::function<Module*()> > factory;
	using id_and_hash = std::pair<int, size_t>;
	boost::container::flat_map<id_and_hash, Module*> modules_map;
public:
	Module *removeModule(int id, size_t type_id);
	void registerType(size_t type_id, boost::function<Module*()> make);
	template<class ...args>
	Module *createModule(int id, size_t type_id, args&&... a) {
		auto idnhash = std::make_pair(id, type_id);
		auto it = modules_map.find(idnhash);
		if (it != modules_map.end())
			return it->second;
		Module* module = factory.create(type_id, a...);
		if (module)
			modules_map.insert(std::make_pair(idnhash, module));
		return module;
	}
};