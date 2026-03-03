import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { ProblemsPanel } from './ProblemsPanel';
import { LogsPanel } from './LogsPanel';
import { usePipelineStore } from '../../store/pipelineStore';
import { useRuntimeStore } from '../../store/runtimeStore';

type TabId = 'problems' | 'logs';

/**
 * Bottom panel with tab bar for switching between Problems and Logs views
 */
export function BottomPanel() {
  const [activeTab, setActiveTab] = useState<TabId>('problems');
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Counts for tab badges
  const validationResult = usePipelineStore((state) => state.validationResult);
  const runtimeErrors = useRuntimeStore((state) => state.errors);
  const logs = useRuntimeStore((state) => state.logs);

  const problemsCount = (validationResult?.issues?.length || 0) + runtimeErrors.length;
  const logsCount = logs.length;

  if (isCollapsed) {
    return (
      <div className="border-t border-gray-200 bg-gray-50">
        <div className="w-full px-4 py-1.5 flex items-center justify-between text-sm text-gray-600">
          <div className="flex items-center gap-3">
            <TabLabel
              label="Problems"
              count={problemsCount}
              active={activeTab === 'problems'}
              onClick={() => { setActiveTab('problems'); setIsCollapsed(false); }}
            />
            <TabLabel
              label="Logs"
              count={logsCount}
              active={activeTab === 'logs'}
              onClick={() => { setActiveTab('logs'); setIsCollapsed(false); }}
            />
          </div>
          <button
            onClick={() => setIsCollapsed(false)}
            className="p-1 text-gray-400 hover:text-gray-600"
            aria-label="Expand panel"
          >
            <ChevronUp className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="border-t border-gray-200 bg-white flex flex-col h-48">
      {/* Tab bar */}
      <div className="flex items-center justify-between px-4 py-1 bg-gray-100 border-b border-gray-200">
        <div className="flex items-center gap-1">
          <TabLabel
            label="Problems"
            count={problemsCount}
            active={activeTab === 'problems'}
            onClick={() => setActiveTab('problems')}
          />
          <TabLabel
            label="Logs"
            count={logsCount}
            active={activeTab === 'logs'}
            onClick={() => setActiveTab('logs')}
          />
        </div>
        <button
          onClick={() => setIsCollapsed(true)}
          className="p-1 text-gray-400 hover:text-gray-600"
          aria-label="Collapse panel"
        >
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>

      {/* Active panel */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'problems' ? <ProblemsPanel /> : <LogsPanel />}
      </div>
    </div>
  );
}

/**
 * Tab label with count badge
 */
function TabLabel({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-xs font-medium rounded-t transition-colors ${
        active
          ? 'bg-white text-gray-900 border border-b-0 border-gray-200'
          : 'text-gray-500 hover:text-gray-700'
      }`}
    >
      {label}
      {count > 0 && (
        <span className={`ml-1.5 px-1.5 py-0.5 text-xs rounded-full ${
          active ? 'bg-gray-200 text-gray-700' : 'bg-gray-300 text-gray-600'
        }`}>
          {count}
        </span>
      )}
    </button>
  );
}
