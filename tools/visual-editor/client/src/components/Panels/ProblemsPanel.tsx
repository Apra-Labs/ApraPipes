import { useState } from 'react';
import { usePipelineStore } from '../../store/pipelineStore';
import { useCanvasStore } from '../../store/canvasStore';
import type { ValidationIssue } from '../../types/validation';
import {
  getSeverityIcon,
  getSeverityColor,
  getSeverityBgColor,
  parseLocationModuleId,
} from '../../types/validation';

/**
 * Filter tabs for the problems panel
 */
type FilterType = 'all' | 'error' | 'warning' | 'info';

/**
 * Individual issue row component
 */
function IssueRow({
  issue,
  onClick,
}: {
  issue: ValidationIssue;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 border-b hover:bg-gray-50 transition-colors ${getSeverityBgColor(issue.level)}`}
    >
      <div className="flex items-start gap-2">
        <span className="flex-shrink-0 mt-0.5" aria-label={issue.level}>
          {getSeverityIcon(issue.level)}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-xs font-mono ${getSeverityColor(issue.level)}`}>
              {issue.code}
            </span>
            <span className="text-sm text-gray-900 truncate">{issue.message}</span>
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
 * Displays validation issues with filtering and click-to-jump
 */
export function ProblemsPanel() {
  const validationResult = usePipelineStore((state) => state.validationResult);
  const isValidating = usePipelineStore((state) => state.isValidating);
  const validate = usePipelineStore((state) => state.validate);
  const selectNode = useCanvasStore((state) => state.selectNode);
  const centerOnNode = useCanvasStore((state) => state.centerOnNode);

  const [filter, setFilter] = useState<FilterType>('all');
  const [isCollapsed, setIsCollapsed] = useState(false);

  const issues = validationResult?.issues || [];

  // Count issues by severity
  const errorCount = issues.filter((i) => i.level === 'error').length;
  const warningCount = issues.filter((i) => i.level === 'warning').length;
  const infoCount = issues.filter((i) => i.level === 'info').length;

  // Filter issues
  const filteredIssues =
    filter === 'all' ? issues : issues.filter((i) => i.level === filter);

  // Handle clicking an issue
  const handleIssueClick = (issue: ValidationIssue) => {
    const moduleId = parseLocationModuleId(issue.location);
    if (moduleId) {
      selectNode(moduleId);
      centerOnNode(moduleId);
    }
  };

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
              count={issues.length}
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
        {!validationResult ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            Click "Validate" to check your pipeline
          </div>
        ) : filteredIssues.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            {validationResult.valid ? (
              <span className="text-green-600">✓ No issues found</span>
            ) : (
              'No issues match the current filter'
            )}
          </div>
        ) : (
          <div>
            {filteredIssues.map((issue, index) => (
              <IssueRow
                key={`${issue.code}-${issue.location}-${index}`}
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
