import { useState, useCallback } from 'react';
import { Download } from 'lucide-react';
import { usePipelineStore } from '../../store/pipelineStore';
import { useCanvasStore } from '../../store/canvasStore';
import { useRuntimeStore } from '../../store/runtimeStore';
import type { ValidationIssue } from '../../types/validation';
import type { RuntimeError } from '../../types/runtime';
import {
  getSeverityIcon,
  getSeverityColor,
  getSeverityBgColor,
  parseLocationModuleId,
} from '../../types/validation';

/**
 * Filter tabs for the problems panel
 */
type FilterType = 'all' | 'error' | 'warning' | 'info' | 'runtime';

/**
 * Extended issue type that includes source (validation or runtime)
 */
interface DisplayIssue extends ValidationIssue {
  /** Source of the issue */
  source: 'validation' | 'runtime';
  /** Original timestamp for runtime errors */
  timestamp?: number;
}

/**
 * Convert a runtime error to a display issue format
 */
function runtimeErrorToDisplayIssue(error: RuntimeError): DisplayIssue {
  return {
    level: 'error',
    code: error.code || 'R001',
    message: error.message,
    location: `modules.${error.moduleId}`,
    suggestion: 'Check module configuration or input data',
    source: 'runtime',
    timestamp: error.timestamp,
  };
}

/**
 * Individual issue row component
 */
function IssueRow({
  issue,
  onClick,
}: {
  issue: DisplayIssue;
  onClick: () => void;
}) {
  // Use purple background for runtime errors
  const bgClass = issue.source === 'runtime'
    ? 'bg-purple-50 border-purple-200'
    : getSeverityBgColor(issue.level);

  // Format timestamp for runtime errors
  const timeString = issue.timestamp
    ? new Date(issue.timestamp).toLocaleTimeString()
    : null;

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 border-b hover:bg-gray-50 transition-colors ${bgClass}`}
    >
      <div className="flex items-start gap-2">
        <span className="flex-shrink-0 mt-0.5" aria-label={issue.level}>
          {issue.source === 'runtime' ? '⚡' : getSeverityIcon(issue.level)}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-xs font-mono ${issue.source === 'runtime' ? 'text-purple-600' : getSeverityColor(issue.level)}`}>
              {issue.code}
            </span>
            {issue.source === 'runtime' && (
              <span className="text-xs bg-purple-100 text-purple-700 px-1 rounded">runtime</span>
            )}
            <span className="text-sm text-gray-900 truncate">{issue.message}</span>
            {timeString && (
              <span className="text-xs text-gray-400 ml-auto">{timeString}</span>
            )}
          </div>
          {issue.location && (
            <div className="text-xs text-gray-500 font-mono mt-0.5 truncate">
              {issue.location}
            </div>
          )}
          {issue.suggestion && (
            <div className="text-xs text-gray-600 mt-1 italic">
              {issue.suggestion}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

/**
 * Filter button component
 */
function FilterButton({
  type,
  label,
  count,
  active,
  onClick,
}: {
  type: FilterType;
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  const getButtonColor = () => {
    if (!active) return 'bg-gray-100 text-gray-600 hover:bg-gray-200';
    switch (type) {
      case 'error':
        return 'bg-red-100 text-red-700';
      case 'warning':
        return 'bg-yellow-100 text-yellow-700';
      case 'info':
        return 'bg-blue-100 text-blue-700';
      case 'runtime':
        return 'bg-purple-100 text-purple-700';
      default:
        return 'bg-gray-200 text-gray-700';
    }
  };

  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 text-xs font-medium rounded transition-colors ${getButtonColor()}`}
    >
      {label} ({count})
    </button>
  );
}

/**
 * Problems Panel component
 * Displays validation issues and runtime errors with filtering and click-to-jump
 */
export function ProblemsPanel() {
  const validationResult = usePipelineStore((state) => state.validationResult);
  const isValidating = usePipelineStore((state) => state.isValidating);
  const validate = usePipelineStore((state) => state.validate);
  const selectNode = useCanvasStore((state) => state.selectNode);
  const centerOnNode = useCanvasStore((state) => state.centerOnNode);

  // Runtime errors from runtimeStore
  const runtimeErrors = useRuntimeStore((state) => state.errors);
  const runtimeStatus = useRuntimeStore((state) => state.status);
  const clearErrors = useRuntimeStore((state) => state.clearErrors);

  const [filter, setFilter] = useState<FilterType>('all');
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Convert validation issues to display format
  const validationIssues: DisplayIssue[] = (validationResult?.issues || []).map((issue) => ({
    ...issue,
    source: 'validation' as const,
  }));

  // Convert runtime errors to display format
  const runtimeIssues: DisplayIssue[] = runtimeErrors.map(runtimeErrorToDisplayIssue);

  // Combine all issues
  const allIssues: DisplayIssue[] = [...validationIssues, ...runtimeIssues];

  // Count issues by type
  const errorCount = validationIssues.filter((i) => i.level === 'error').length;
  const warningCount = validationIssues.filter((i) => i.level === 'warning').length;
  const infoCount = validationIssues.filter((i) => i.level === 'info').length;
  const runtimeCount = runtimeIssues.length;

  // Filter issues based on selected filter
  const filteredIssues = (() => {
    switch (filter) {
      case 'all':
        return allIssues;
      case 'runtime':
        return runtimeIssues;
      case 'error':
        return validationIssues.filter((i) => i.level === 'error');
      case 'warning':
        return validationIssues.filter((i) => i.level === 'warning');
      case 'info':
        return validationIssues.filter((i) => i.level === 'info');
      default:
        return allIssues;
    }
  })();

  // Handle clicking an issue
  const handleIssueClick = (issue: DisplayIssue) => {
    const moduleId = parseLocationModuleId(issue.location);
    if (moduleId) {
      selectNode(moduleId);
      centerOnNode(moduleId);
    }
  };

  // Export logs functionality
  const handleExportLogs = useCallback(() => {
    const logs = {
      exportedAt: new Date().toISOString(),
      pipelineStatus: runtimeStatus,
      validationIssues: validationResult?.issues || [],
      runtimeErrors: runtimeErrors.map((e) => ({
        ...e,
        timestampFormatted: new Date(e.timestamp).toISOString(),
      })),
    };

    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [runtimeStatus, validationResult, runtimeErrors]);

  if (isCollapsed) {
    return (
      <div className="border-t border-gray-200 bg-gray-50">
        <button
          onClick={() => setIsCollapsed(false)}
          className="w-full px-4 py-2 flex items-center justify-between text-sm text-gray-600 hover:bg-gray-100"
        >
          <span className="font-medium">Problems</span>
          <div className="flex items-center gap-2">
            {errorCount > 0 && (
              <span className="px-1.5 py-0.5 text-xs bg-red-100 text-red-700 rounded">
                {errorCount} errors
              </span>
            )}
            {warningCount > 0 && (
              <span className="px-1.5 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">
                {warningCount} warnings
              </span>
            )}
            {runtimeCount > 0 && (
              <span className="px-1.5 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">
                {runtimeCount} runtime
              </span>
            )}
            <span>▲</span>
          </div>
        </button>
      </div>
    );
  }

  return (
    <div className="border-t border-gray-200 bg-white flex flex-col h-48">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <h3 className="font-medium text-sm text-gray-700">Problems</h3>
          <div className="flex items-center gap-1">
            <FilterButton
              type="all"
              label="All"
              count={allIssues.length}
              active={filter === 'all'}
              onClick={() => setFilter('all')}
            />
            <FilterButton
              type="error"
              label="Errors"
              count={errorCount}
              active={filter === 'error'}
              onClick={() => setFilter('error')}
            />
            <FilterButton
              type="warning"
              label="Warnings"
              count={warningCount}
              active={filter === 'warning'}
              onClick={() => setFilter('warning')}
            />
            <FilterButton
              type="info"
              label="Info"
              count={infoCount}
              active={filter === 'info'}
              onClick={() => setFilter('info')}
            />
            <FilterButton
              type="runtime"
              label="Runtime"
              count={runtimeCount}
              active={filter === 'runtime'}
              onClick={() => setFilter('runtime')}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => validate()}
            disabled={isValidating}
            className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isValidating ? 'Validating...' : 'Validate'}
          </button>
          {(runtimeErrors.length > 0 || validationResult) && (
            <button
              onClick={handleExportLogs}
              className="px-2 py-1 text-xs bg-gray-500 text-white rounded hover:bg-gray-600 flex items-center gap-1"
              title="Export logs"
            >
              <Download className="w-3 h-3" />
              Export
            </button>
          )}
          {runtimeErrors.length > 0 && (
            <button
              onClick={clearErrors}
              className="px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
              title="Clear runtime errors"
            >
              Clear
            </button>
          )}
          <button
            onClick={() => setIsCollapsed(true)}
            className="p-1 text-gray-500 hover:text-gray-700"
            aria-label="Collapse panel"
          >
            ▼
          </button>
        </div>
      </div>

      {/* Issues list */}
      <div className="flex-1 overflow-y-auto">
        {!validationResult && runtimeErrors.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            Click "Validate" to check your pipeline
          </div>
        ) : filteredIssues.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            {filter === 'runtime' ? (
              'No runtime errors'
            ) : validationResult?.valid && runtimeErrors.length === 0 ? (
              <span className="text-green-600">✓ No issues found</span>
            ) : (
              'No issues match the current filter'
            )}
          </div>
        ) : (
          <div>
            {filteredIssues.map((issue, index) => (
              <IssueRow
                key={`${issue.source}-${issue.code}-${issue.location}-${index}`}
                issue={issue}
                onClick={() => handleIssueClick(issue)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
