import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { ModuleNodeData } from '../../store/canvasStore';
import { getCategoryColor } from '../../types/schema';

/**
 * Tooltip component for showing pin information
 */
function PinTooltip({ name, frameTypes, position }: { name: string; frameTypes: string[]; position: 'left' | 'right' }) {
  return (
    <div
      className={`absolute z-50 px-2 py-1 text-xs bg-gray-900 text-white rounded shadow-lg whitespace-nowrap ${
        position === 'left' ? 'right-full mr-2' : 'left-full ml-2'
      }`}
      style={{ top: '50%', transform: 'translateY(-50%)' }}
    >
      <div className="font-semibold">{name}</div>
      {frameTypes.length > 0 && (
        <div className="text-gray-300 text-[10px]">{frameTypes.join(', ')}</div>
      )}
    </div>
  );
}

interface ModuleNodeProps {
  data: ModuleNodeData;
  selected?: boolean;
}

/**
 * Custom node component for rendering pipeline modules
 */
function ModuleNodeComponent({ data, selected }: ModuleNodeProps) {
  const categoryColorClass = getCategoryColor(data.category);
  const [hoveredPin, setHoveredPin] = useState<string | null>(null);

  const statusColors: Record<string, string> = {
    idle: 'border-gray-300',
    running: 'border-green-500 shadow-green-200',
    error: 'border-red-500 shadow-red-200',
  };

  return (
    <div
      className={`
        bg-white rounded-lg shadow-md border-2 min-w-[180px]
        ${statusColors[data.status] || statusColors.idle}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-1' : ''}
        ${data.status === 'running' ? 'shadow-lg' : ''}
      `}
    >
      {/* Header */}
      <div className={`px-3 py-2 rounded-t-md ${categoryColorClass}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium uppercase opacity-80">
            {data.category}
          </span>
          {data.status === 'error' && (
            <span className="text-xs">⚠️</span>
          )}
        </div>
        <h3 className="font-semibold text-sm truncate" title={data.type}>
          {data.label || data.type}
        </h3>
      </div>

      {/* Metrics (shown when running) */}
      {data.metrics && data.status === 'running' && (
        <div className="px-3 py-1 bg-gray-50 text-xs border-b border-gray-200 flex gap-3">
          <span className="text-gray-600">
            FPS: <span className="font-medium text-gray-800">{data.metrics.fps}</span>
          </span>
          <span className="text-gray-600">
            Q: <span className={`font-medium ${data.metrics.isQueueFull ? 'text-red-500' : 'text-gray-800'}`}>
              {data.metrics.qlen}
            </span>
          </span>
        </div>
      )}

      {/* Body - Pins */}
      <div className="relative px-2 py-3">
        {/* Input pins (left side) */}
        <div className="space-y-2">
          {data.inputs.map((input: { name: string; frame_types: string[] }, index: number) => (
            <div
              key={input.name}
              className="flex items-center gap-2 relative"
              onMouseEnter={() => setHoveredPin(`in-${input.name}`)}
              onMouseLeave={() => setHoveredPin(null)}
            >
              <Handle
                type="target"
                position={Position.Left}
                id={input.name}
                className="!w-3 !h-3 !bg-blue-500 !border-2 !border-white hover:!bg-blue-600 hover:!scale-125 transition-transform"
                style={{ top: `${24 + index * 24}px` }}
              />
              <span className="text-xs text-gray-500 pl-2">{input.name}</span>
              {hoveredPin === `in-${input.name}` && (
                <PinTooltip name={input.name} frameTypes={input.frame_types} position="left" />
              )}
            </div>
          ))}
        </div>

        {/* Output pins (right side) */}
        <div className="space-y-2">
          {data.outputs.map((output: { name: string; frame_types: string[] }, index: number) => (
            <div
              key={output.name}
              className="flex items-center justify-end gap-2 relative"
              onMouseEnter={() => setHoveredPin(`out-${output.name}`)}
              onMouseLeave={() => setHoveredPin(null)}
            >
              <span className="text-xs text-gray-500 pr-2">{output.name}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={output.name}
                className="!w-3 !h-3 !bg-green-500 !border-2 !border-white hover:!bg-green-600 hover:!scale-125 transition-transform"
                style={{ top: `${24 + index * 24}px` }}
              />
              {hoveredPin === `out-${output.name}` && (
                <PinTooltip name={output.name} frameTypes={output.frame_types} position="right" />
              )}
            </div>
          ))}
        </div>

        {/* Show placeholder if no pins */}
        {data.inputs.length === 0 && data.outputs.length === 0 && (
          <div className="text-xs text-gray-400 text-center py-1">
            No pins defined
          </div>
        )}
      </div>
    </div>
  );
}

export const ModuleNode = memo(ModuleNodeComponent);
