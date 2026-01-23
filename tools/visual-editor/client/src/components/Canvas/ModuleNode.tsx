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

  const hasErrors = (data.validationErrors ?? 0) > 0;
  const hasWarnings = (data.validationWarnings ?? 0) > 0;

  // Determine border color based on status and validation
  const getBorderClass = () => {
    if (hasErrors) return 'border-red-500 shadow-red-100';
    if (hasWarnings) return 'border-yellow-500 shadow-yellow-100';
    if (data.status === 'running') return 'border-green-500 shadow-green-200';
    if (data.status === 'error') return 'border-red-500 shadow-red-200';
    return 'border-gray-300';
  };

  return (
    <div
      className={`
        bg-white rounded-lg shadow-md border-2 min-w-[180px]
        ${getBorderClass()}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-1' : ''}
        ${data.status === 'running' || hasErrors ? 'shadow-lg' : ''}
      `}
    >
      {/* Header */}
      <div className={`px-3 py-2 rounded-t-md ${categoryColorClass}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium uppercase opacity-80">
            {data.category}
          </span>
          <div className="flex items-center gap-1">
            {hasErrors && (
              <span
                className="inline-flex items-center justify-center w-5 h-5 text-xs bg-red-500 text-white rounded-full"
                title={`${data.validationErrors} error(s)`}
              >
                {data.validationErrors}
              </span>
            )}
            {hasWarnings && (
              <span
                className="inline-flex items-center justify-center w-5 h-5 text-xs bg-yellow-500 text-white rounded-full"
                title={`${data.validationWarnings} warning(s)`}
              >
                {data.validationWarnings}
              </span>
            )}
            {data.status === 'error' && !hasErrors && (
              <span className="text-xs">⚠️</span>
            )}
          </div>
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
