import { useCallback, useState } from 'react';
import { Trash2, ArrowRight, ArrowLeft } from 'lucide-react';
import { useCanvasStore } from '../../store/canvasStore';
import { usePipelineStore } from '../../store/pipelineStore';
import type { PropertySchema, ModuleSchema } from '../../types/schema';
import {
  IntInput,
  FloatInput,
  BoolCheckbox,
  StringInput,
  EnumDropdown,
  JsonEditor,
  PathInput,
} from './PropertyEditors';

interface PropertyPanelProps {
  schema: Record<string, ModuleSchema> | null;
}

/**
 * Property panel for editing selected module properties
 */
export function PropertyPanel({ schema }: PropertyPanelProps) {
  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const selectedPin = useCanvasStore((state) => state.selectedPin);
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const updateNodeData = useCanvasStore((state) => state.updateNodeData);
  const removeNode = useCanvasStore((state) => state.removeNode);
  const centerOnNode = useCanvasStore((state) => state.centerOnNode);
  const selectNode = useCanvasStore((state) => state.selectNode);

  const updateModuleProperty = usePipelineStore((state) => state.updateModuleProperty);
  const pipelineConfig = usePipelineStore((state) => state.config);
  const renameModule = usePipelineStore((state) => state.renameModule);
  const removeModule = usePipelineStore((state) => state.removeModule);

  const [editingName, setEditingName] = useState(false);
  const [tempName, setTempName] = useState('');

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const moduleSchema = selectedNode && schema ? schema[selectedNode.data.type] : null;
  const moduleConfig = selectedNodeId ? pipelineConfig.modules[selectedNodeId] : null;

  const handlePropertyChange = useCallback(
    (key: string, value: unknown) => {
      if (!selectedNodeId) return;
      updateModuleProperty(selectedNodeId, key, value);
    },
    [selectedNodeId, updateModuleProperty]
  );

  const handleNameEdit = useCallback(() => {
    if (selectedNode) {
      setTempName(selectedNode.data.label);
      setEditingName(true);
    }
  }, [selectedNode]);

  const handleNameSave = useCallback(() => {
    if (!selectedNodeId || !tempName.trim()) {
      setEditingName(false);
      return;
    }

    const newId = tempName.trim();
    if (newId !== selectedNodeId) {
      // Update canvas store
      updateNodeData(selectedNodeId, { label: newId });
      // Update pipeline store
      renameModule(selectedNodeId, newId);
    }
    setEditingName(false);
  }, [selectedNodeId, tempName, updateNodeData, renameModule]);

  const handleNameKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleNameSave();
      } else if (e.key === 'Escape') {
        setEditingName(false);
      }
    },
    [handleNameSave]
  );

  const handleDelete = useCallback(() => {
    if (!selectedNodeId) return;
    const confirmed = window.confirm(`Delete module "${selectedNode?.data.label || selectedNodeId}"?`);
    if (confirmed) {
      removeNode(selectedNodeId);
      removeModule(selectedNodeId);
    }
  }, [selectedNodeId, selectedNode, removeNode, removeModule]);

  const getPropertyValue = (key: string, propSchema: PropertySchema): unknown => {
    if (moduleConfig?.properties?.[key] !== undefined) {
      return moduleConfig.properties[key];
    }
    // Return default from schema
    if (propSchema.default !== undefined) {
      switch (propSchema.type) {
        case 'int':
          return parseInt(propSchema.default, 10);
        case 'float':
          return parseFloat(propSchema.default);
        case 'bool':
          return propSchema.default === 'true';
        case 'json':
          try {
            return JSON.parse(propSchema.default);
          } catch {
            return {};
          }
        default:
          return propSchema.default;
      }
    }
    // Return type-appropriate default
    switch (propSchema.type) {
      case 'int':
        return 0;
      case 'float':
        return 0.0;
      case 'bool':
        return false;
      case 'string':
        return '';
      case 'enum':
        return propSchema.enum_values?.[0] ?? '';
      case 'json':
        return {};
      default:
        return '';
    }
  };

  // Detect if a string property is likely a file/path
  const isPathProperty = (key: string, description?: string): boolean => {
    const pathKeywords = ['path', 'file', 'dir', 'folder', 'directory', 'location'];
    const keyLower = key.toLowerCase();
    const descLower = (description || '').toLowerCase();
    return pathKeywords.some((kw) => keyLower.includes(kw) || descLower.includes(kw));
  };

  const renderPropertyEditor = (key: string, propSchema: PropertySchema) => {
    const value = getPropertyValue(key, propSchema);

    switch (propSchema.type) {
      case 'int':
        return (
          <IntInput
            key={key}
            label={key}
            value={value as number}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      case 'float':
        return (
          <FloatInput
            key={key}
            label={key}
            value={value as number}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      case 'bool':
        return (
          <BoolCheckbox
            key={key}
            label={key}
            value={value as boolean}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      case 'string':
        // Use PathInput for path-like properties
        if (isPathProperty(key, propSchema.description)) {
          return (
            <PathInput
              key={key}
              label={key}
              value={value as string}
              schema={propSchema}
              onChange={(v) => handlePropertyChange(key, v)}
            />
          );
        }
        return (
          <StringInput
            key={key}
            label={key}
            value={value as string}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      case 'enum':
        return (
          <EnumDropdown
            key={key}
            label={key}
            value={value as string}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      case 'json':
        return (
          <JsonEditor
            key={key}
            label={key}
            value={value}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
      default:
        return (
          <StringInput
            key={key}
            label={key}
            value={String(value)}
            schema={propSchema}
            onChange={(v) => handlePropertyChange(key, v)}
          />
        );
    }
  };

  // Get pin information if a pin is selected
  const getPinInfo = () => {
    if (!selectedPin) return null;

    const pinNode = nodes.find((n) => n.id === selectedPin.nodeId);
    if (!pinNode) return null;

    const pins = selectedPin.pinType === 'input' ? pinNode.data.inputs : pinNode.data.outputs;
    const pin = pins.find((p: { name: string; frame_types: string[] }) => p.name === selectedPin.pinName);
    if (!pin) return null;

    // Find connected edges
    const connectedEdges = edges.filter((e) => {
      if (selectedPin.pinType === 'input') {
        return e.target === selectedPin.nodeId && e.targetHandle === selectedPin.pinName;
      } else {
        return e.source === selectedPin.nodeId && e.sourceHandle === selectedPin.pinName;
      }
    });

    // Find connected modules
    const connectedModules = connectedEdges.map((e) => {
      if (selectedPin.pinType === 'input') {
        const sourceNode = nodes.find((n) => n.id === e.source);
        return { nodeId: e.source, label: sourceNode?.data.label || e.source, pinName: e.sourceHandle || 'default' };
      } else {
        const targetNode = nodes.find((n) => n.id === e.target);
        return { nodeId: e.target, label: targetNode?.data.label || e.target, pinName: e.targetHandle || 'default' };
      }
    });

    return {
      ...pin,
      direction: selectedPin.pinType,
      moduleName: pinNode.data.label,
      moduleType: pinNode.data.type,
      connectedModules,
    };
  };

  const pinInfo = getPinInfo();

  // Show pin properties if a pin is selected
  if (selectedPin && pinInfo) {
    return (
      <div className="p-4 space-y-4">
        <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
          Pin Properties
        </h3>

        {/* Pin Name */}
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Pin Name</label>
          <div className="flex items-center gap-2 px-2 py-1 text-sm bg-gray-50 border border-gray-200 rounded">
            {pinInfo.direction === 'input' ? (
              <ArrowRight className="w-4 h-4 text-blue-500" />
            ) : (
              <ArrowLeft className="w-4 h-4 text-green-500" />
            )}
            <span className="font-mono">{pinInfo.name}</span>
          </div>
        </div>

        {/* Direction */}
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Direction</label>
          <div className={`inline-flex items-center px-2 py-1 text-xs rounded-full ${
            pinInfo.direction === 'input'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-green-100 text-green-700'
          }`}>
            {pinInfo.direction === 'input' ? 'Input (Target)' : 'Output (Source)'}
          </div>
        </div>

        {/* Frame Types */}
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Supported Frame Types</label>
          {pinInfo.frame_types.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {pinInfo.frame_types.map((ft: string) => (
                <span
                  key={ft}
                  className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
                >
                  {ft}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">Any frame type</p>
          )}
        </div>

        {/* Parent Module */}
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Parent Module</label>
          <button
            onClick={() => {
              selectNode(selectedPin.nodeId);
            }}
            className="w-full text-left px-2 py-1 text-sm bg-gray-50 border border-gray-200 rounded hover:bg-gray-100 transition-colors"
          >
            <span className="font-medium">{pinInfo.moduleName}</span>
            <span className="text-gray-400 ml-1">({pinInfo.moduleType})</span>
          </button>
        </div>

        {/* Connected Modules */}
        <div className="border-t border-gray-200 pt-4">
          <label className="block text-xs font-medium text-gray-600 mb-2">
            {pinInfo.direction === 'input' ? 'Connected From' : 'Connected To'}
          </label>
          {pinInfo.connectedModules.length > 0 ? (
            <div className="space-y-1">
              {pinInfo.connectedModules.map((conn: { nodeId: string; label: string; pinName: string }) => (
                <button
                  key={`${conn.nodeId}-${conn.pinName}`}
                  onClick={() => {
                    selectNode(conn.nodeId);
                    centerOnNode(conn.nodeId);
                  }}
                  className="w-full flex items-center justify-between px-2 py-1.5 text-sm bg-gray-50 border border-gray-200 rounded hover:bg-blue-50 hover:border-blue-300 transition-colors"
                >
                  <span>{conn.label}</span>
                  <span className="text-xs text-gray-400 font-mono">.{conn.pinName}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">Not connected</p>
          )}
        </div>
      </div>
    );
  }

  if (!selectedNode) {
    return (
      <div className="p-4">
        <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide mb-4">
          Properties
        </h3>
        <p className="text-sm text-muted-foreground">
          Select a module or pin to view its properties
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
          Properties
        </h3>
        <button
          onClick={handleDelete}
          className="p-1.5 text-red-500 hover:text-red-700 hover:bg-red-50 rounded transition-colors"
          title="Delete module (Delete key)"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Module Name (Editable) */}
      <div className="space-y-1">
        <label className="block text-xs font-medium text-gray-600">Module Name</label>
        {editingName ? (
          <input
            type="text"
            value={tempName}
            onChange={(e) => setTempName(e.target.value)}
            onBlur={handleNameSave}
            onKeyDown={handleNameKeyDown}
            autoFocus
            className="w-full px-2 py-1 text-sm border border-blue-500 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        ) : (
          <div
            onClick={handleNameEdit}
            className="w-full px-2 py-1 text-sm bg-gray-50 border border-gray-200 rounded cursor-pointer hover:bg-gray-100"
          >
            {selectedNode.data.label}
          </div>
        )}
      </div>

      {/* Module Type (Read-only) */}
      <div className="space-y-1">
        <label className="block text-xs font-medium text-gray-600">Module Type</label>
        <div className="px-2 py-1 text-sm bg-muted rounded font-mono">
          {selectedNode.data.type}
        </div>
      </div>

      {/* Category */}
      <div className="space-y-1">
        <label className="block text-xs font-medium text-gray-600">Category</label>
        <div className="text-sm capitalize">{selectedNode.data.category}</div>
      </div>

      {/* Description */}
      {selectedNode.data.description && (
        <div className="space-y-1">
          <label className="block text-xs font-medium text-gray-600">Description</label>
          <p className="text-sm text-gray-600">{selectedNode.data.description}</p>
        </div>
      )}

      {/* Divider */}
      {moduleSchema && Object.keys(moduleSchema.properties).length > 0 && (
        <div className="border-t border-gray-200 pt-4">
          <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
            Configuration
          </h4>
          <div className="space-y-3">
            {Object.entries(moduleSchema.properties).map(([key, propSchema]) =>
              renderPropertyEditor(key, propSchema)
            )}
          </div>
        </div>
      )}

      {/* Module ID (for debugging) */}
      <div className="border-t border-gray-200 pt-4">
        <label className="block text-xs font-medium text-gray-400">ID</label>
        <p className="font-mono text-xs text-gray-400 break-all">{selectedNode.id}</p>
      </div>
    </div>
  );
}
