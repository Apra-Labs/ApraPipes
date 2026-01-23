import { memo, useState, useCallback } from 'react';
import { Handle, Position, useNodeId } from '@xyflow/react';
import type { ModuleNodeData } from '../../store/canvasStore';
import { useCanvasStore } from '../../store/canvasStore';
import { useRuntimeStore } from '../../store/runtimeStore';
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
 * Status indicator badge component
 */
function StatusBadge({ status }: { status: 'idle' | 'running' | 'error' }) {
  const colors = {
    idle: 'bg-gray-400',
    running: 'bg-green-500 animate-pulse',
    error: 'bg-red-500',
  };

  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${colors[status]}`}
      title={status.charAt(0).toUpperCase() + status.slice(1)}
    />
  );
}

/**
 * Custom node component for rendering pipeline modules
 */
function ModuleNodeComponent({ data, selected }: ModuleNodeProps) {
  const categoryColorClass = getCategoryColor(data.category);
  const [hoveredPin, setHoveredPin] = useState<string | null>(null);
  const nodeId = useNodeId();

  // Pin selection
  const selectPin = useCanvasStore((state) => state.selectPin);
  const selectedPin = useCanvasStore((state) => state.selectedPin);

  const handlePinClick = useCallback((pinName: string, pinType: 'input' | 'output', e: React.MouseEvent) => {
    e.stopPropagation(); // Don't trigger node selection
    if (nodeId) {
      selectPin({ nodeId, pinName, pinType });
    }
  }, [nodeId, selectPin]);

  const isPinSelected = (pinName: string, pinType: 'input' | 'output') => {
    return selectedPin?.nodeId === nodeId && selectedPin?.pinName === pinName && selectedPin?.pinType === pinType;
  };

  // Get runtime metrics from store
  const runtimeStatus = useRuntimeStore((state) => state.status);
  const runtimeMetrics = useRuntimeStore((state) => state.moduleMetrics[data.label] || null);

  // Determine effective status (use runtime status if running, else data status)
  const isRunning = runtimeStatus === 'RUNNING';
  const effectiveStatus = isRunning ? 'running' : data.status;
  const metrics = isRunning ? runtimeMetrics : data.metrics;

  const hasErrors = (data.validationErrors ?? 0) > 0;
  const hasWarnings = (data.validationWarnings ?? 0) > 0;

  // Determine border color based on status and validation
  const getBorderClass = () => {
    if (hasErrors) return 'border-red-500 shadow-red-100';
    if (hasWarnings) return 'border-yellow-500 shadow-yellow-100';
    if (effectiveStatus === 'running') return 'border-green-500 shadow-green-200';
    if (effectiveStatus === 'error') return 'border-red-500 shadow-red-200';
    return 'border-gray-300';
  };

  return (
    <div
      className={`
        bg-white rounded-lg shadow-md border-2 min-w-[180px]
        ${getBorderClass()}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-1' : ''}
        ${effectiveStatus === 'running' || hasErrors ? 'shadow-lg' : ''}
      `}
    >
      {/* Header - compact design, category shown by color only */}
      <div className={`px-3 py-2 rounded-t-md ${categoryColorClass}`}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <StatusBadge status={effectiveStatus} />
            <h3 className="font-semibold text-sm truncate" title={`${data.type} (${data.category})`}>
              {data.label || data.type}
            </h3>
          </div>
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
            {effectiveStatus === 'error' && !hasErrors && (
              <span className="text-xs">!!</span>
            )}
          </div>
        </div>
      </div>

      {/* Metrics (shown when running) */}
      {metrics && effectiveStatus === 'running' && (
        <div className="px-3 py-1.5 bg-gray-50 text-xs border-b border-gray-200">
          <div className="flex justify-between items-center gap-2">
            <span className="text-gray-600">
              <span className="font-medium text-gray-800">{metrics.fps.toFixed(1)}</span> fps
            </span>
            <span className="text-gray-600">
              Queue: <span className={`font-medium ${metrics.isQueueFull ? 'text-red-500' : 'text-gray-800'}`}>
                {metrics.qlen}
              </span>
            </span>
          </div>
          {/* Queue fill progress bar */}
          <div className="mt-1 h-1 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                metrics.isQueueFull ? 'bg-red-500' : metrics.qlen > 5 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${Math.min(metrics.qlen * 10, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Body - Pins */}
      <div className="relative px-2 py-3">
        {/* Input pins (left side) */}
        <div className="space-y-2">
          {data.inputs.map((input: { name: string; frame_types: string[] }, index: number) => (
            <div
              key={input.name}
              className={`flex items-center gap-2 relative cursor-pointer rounded px-1 -mx-1 ${
                isPinSelected(input.name, 'input') ? 'bg-blue-100' : 'hover:bg-gray-100'
              }`}
              onMouseEnter={() => setHoveredPin(`in-${input.name}`)}
              onMouseLeave={() => setHoveredPin(null)}
              onClick={(e) => handlePinClick(input.name, 'input', e)}
            >
              <Handle
                type="target"
                position={Position.Left}
                id={input.name}
                className={`!w-4 !h-4 !border-2 !border-white hover:!scale-150 hover:!ring-4 hover:!ring-blue-300/50 transition-all cursor-crosshair ${
                  isPinSelected(input.name, 'input') ? '!bg-blue-700 !ring-2 !ring-blue-500' : '!bg-blue-500 hover:!bg-blue-600'
                }`}
                style={{ top: `${24 + index * 24}px` }}
              />
              <span className={`text-xs pl-2 ${isPinSelected(input.name, 'input') ? 'text-blue-700 font-medium' : 'text-gray-500'}`}>
                {input.name}
              </span>
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
              className={`flex items-center justify-end gap-2 relative cursor-pointer rounded px-1 -mx-1 ${
                isPinSelected(output.name, 'output') ? 'bg-green-100' : 'hover:bg-gray-100'
              }`}
              onMouseEnter={() => setHoveredPin(`out-${output.name}`)}
              onMouseLeave={() => setHoveredPin(null)}
              onClick={(e) => handlePinClick(output.name, 'output', e)}
            >
              <span className={`text-xs pr-2 ${isPinSelected(output.name, 'output') ? 'text-green-700 font-medium' : 'text-gray-500'}`}>
                {output.name}
              </span>
              <Handle
                type="source"
                position={Position.Right}
                id={output.name}
                className={`!w-4 !h-4 !border-2 !border-white hover:!scale-150 hover:!ring-4 hover:!ring-green-300/50 transition-all cursor-crosshair ${
                  isPinSelected(output.name, 'output') ? '!bg-green-700 !ring-2 !ring-green-500' : '!bg-green-500 hover:!bg-green-600'
                }`}
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
