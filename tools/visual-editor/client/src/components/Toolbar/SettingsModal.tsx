import { useCallback } from 'react';
import { X, LayoutGrid } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import { useCanvasStore } from '../../store/canvasStore';

interface SettingsModalProps {
  onClose: () => void;
}

/**
 * Settings modal for configuring editor preferences
 */
export function SettingsModal({ onClose }: SettingsModalProps) {
  const validateOnSave = useUIStore((state) => state.settings.validateOnSave);
  const setValidateOnSave = useUIStore((state) => state.setValidateOnSave);

  const nodes = useCanvasStore((state) => state.nodes);
  const updateNodePosition = useCanvasStore((state) => state.updateNodePosition);
  const saveSnapshot = useCanvasStore((state) => state.saveSnapshot);

  // Auto-arrange nodes in a grid layout
  const handleAutoArrange = useCallback(() => {
    if (nodes.length === 0) return;

    const GRID_SPACING_X = 280;
    const GRID_SPACING_Y = 200;
    const COLS = Math.ceil(Math.sqrt(nodes.length));
    const START_X = 100;
    const START_Y = 100;

    // Sort nodes: sources first, then transforms, then sinks
    const sortedNodes = [...nodes].sort((a, b) => {
      const categoryOrder: Record<string, number> = {
        source: 0,
        transform: 1,
        sink: 2,
      };
      const orderA = categoryOrder[a.data.category] ?? 1;
      const orderB = categoryOrder[b.data.category] ?? 1;
      return orderA - orderB;
    });

    // Position nodes in a grid
    sortedNodes.forEach((node, index) => {
      const col = index % COLS;
      const row = Math.floor(index / COLS);
      const x = START_X + col * GRID_SPACING_X;
      const y = START_Y + row * GRID_SPACING_Y;
      updateNodePosition(node.id, { x, y });
    });

    // Save snapshot for undo
    saveSnapshot('Auto-arrange nodes');
  }, [nodes, updateNodePosition, saveSnapshot]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg shadow-xl w-96">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6">
          {/* Editor Settings Section */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Editor</h3>
            <div className="space-y-3">
              {/* Validate on Save Toggle */}
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-sm text-gray-600">Validate on save</span>
                <button
                  role="switch"
                  aria-checked={validateOnSave}
                  onClick={() => setValidateOnSave(!validateOnSave)}
                  className={`
                    relative w-11 h-6 rounded-full transition-colors
                    ${validateOnSave ? 'bg-blue-500' : 'bg-gray-300'}
                  `}
                >
                  <span
                    className={`
                      absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform
                      ${validateOnSave ? 'translate-x-5' : 'translate-x-0'}
                    `}
                  />
                </button>
              </label>
              <p className="text-xs text-gray-400 ml-0.5">
                Automatically validate the pipeline before saving
              </p>
            </div>
          </div>

          {/* Canvas Actions Section */}
          <div className="border-t border-gray-200 pt-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Canvas</h3>
            <div className="space-y-2">
              {/* Auto-arrange Button */}
              <button
                onClick={handleAutoArrange}
                disabled={nodes.length === 0}
                className={`
                  w-full flex items-center justify-center gap-2 px-4 py-2
                  text-sm font-medium rounded border border-gray-300
                  ${nodes.length === 0
                    ? 'opacity-50 cursor-not-allowed bg-gray-50'
                    : 'hover:bg-gray-50 bg-white'
                  }
                `}
              >
                <LayoutGrid className="w-4 h-4" />
                Auto-arrange Modules
              </button>
              <p className="text-xs text-gray-400 ml-0.5">
                Automatically arrange modules in a grid layout (sources → transforms → sinks)
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 bg-gray-50 rounded-b-lg">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 text-sm font-medium text-white bg-blue-500 hover:bg-blue-600 rounded"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
