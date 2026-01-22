import { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Search } from 'lucide-react';
import type { ModuleSchema } from '../../types/schema';
import { getCategoryColor } from '../../types/schema';

interface ModulePaletteProps {
  modules: Record<string, ModuleSchema>;
}

/**
 * Module palette component for displaying available modules grouped by category
 */
export function ModulePalette({ modules }: ModulePaletteProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['source', 'transform', 'sink'])
  );

  // Group modules by category
  const groupedModules = useMemo(() => {
    const groups: Record<string, Array<{ name: string; schema: ModuleSchema }>> = {};

    Object.entries(modules).forEach(([name, schema]) => {
      const category = schema.category || 'other';
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push({ name, schema });
    });

    // Sort modules within each category
    Object.values(groups).forEach((group) =>
      group.sort((a, b) => a.name.localeCompare(b.name))
    );

    return groups;
  }, [modules]);

  // Filter modules by search query
  const filteredGroups = useMemo(() => {
    if (!searchQuery.trim()) {
      return groupedModules;
    }

    const query = searchQuery.toLowerCase();
    const filtered: Record<string, Array<{ name: string; schema: ModuleSchema }>> = {};

    Object.entries(groupedModules).forEach(([category, moduleList]) => {
      const matches = moduleList.filter(
        ({ name, schema }) =>
          name.toLowerCase().includes(query) ||
          schema.description?.toLowerCase().includes(query)
      );
      if (matches.length > 0) {
        filtered[category] = matches;
      }
    });

    return filtered;
  }, [groupedModules, searchQuery]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  // Sort categories for consistent display order
  const sortedCategories = useMemo(() => {
    const order = ['source', 'transform', 'sink', 'cuda', 'other'];
    return Object.keys(filteredGroups).sort((a, b) => {
      const aIndex = order.indexOf(a);
      const bIndex = order.indexOf(b);
      if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
      if (aIndex === -1) return 1;
      if (bIndex === -1) return -1;
      return aIndex - bIndex;
    });
  }, [filteredGroups]);

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-2 border-b border-border">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search modules..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-sm border border-input rounded bg-background focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
      </div>

      {/* Module List */}
      <div className="flex-1 overflow-y-auto">
        {sortedCategories.map((category) => (
          <CategoryGroup
            key={category}
            category={category}
            modules={filteredGroups[category]}
            expanded={expandedCategories.has(category)}
            onToggle={() => toggleCategory(category)}
          />
        ))}
        {sortedCategories.length === 0 && (
          <div className="p-4 text-center text-muted-foreground text-sm">
            No modules found
          </div>
        )}
      </div>
    </div>
  );
}

interface CategoryGroupProps {
  category: string;
  modules: Array<{ name: string; schema: ModuleSchema }>;
  expanded: boolean;
  onToggle: () => void;
}

function CategoryGroup({ category, modules, expanded, onToggle }: CategoryGroupProps) {
  return (
    <div className="border-b border-border">
      {/* Category Header */}
      <button
        className="w-full px-3 py-2 flex items-center gap-2 hover:bg-muted/50 transition-colors"
        onClick={onToggle}
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground" />
        )}
        <span className="font-medium text-sm capitalize">{category}</span>
        <span className="text-xs text-muted-foreground ml-auto">
          {modules.length}
        </span>
      </button>

      {/* Module List */}
      {expanded && (
        <div className="pb-2">
          {modules.map(({ name, schema }) => (
            <ModuleCard key={name} name={name} schema={schema} />
          ))}
        </div>
      )}
    </div>
  );
}

interface ModuleCardProps {
  name: string;
  schema: ModuleSchema;
}

function ModuleCard({ name, schema }: ModuleCardProps) {
  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/aprapipes-module', name);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      className="mx-2 my-1 p-2 bg-background border border-border rounded cursor-grab hover:border-primary hover:shadow-sm transition-all active:cursor-grabbing"
      title={schema.description}
    >
      <div className="flex items-center gap-2">
        <span
          className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${getCategoryColor(
            schema.category
          )}`}
        >
          {schema.category.slice(0, 3).toUpperCase()}
        </span>
        <span className="text-sm font-medium truncate">{name}</span>
      </div>
      {schema.description && (
        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
          {schema.description}
        </p>
      )}
    </div>
  );
}
