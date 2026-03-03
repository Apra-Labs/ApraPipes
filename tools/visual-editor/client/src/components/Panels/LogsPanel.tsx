import { useState, useRef, useEffect, useCallback } from 'react';
import { Download, Trash2, ArrowDownToLine } from 'lucide-react';
import { useRuntimeStore } from '../../store/runtimeStore';
import type { LogLevel, LogEntry } from '../../types/runtime';

/**
 * Format a timestamp as HH:MM:SS.mmm
 */
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

/**
 * Get display styling for a log level
 */
function getLevelStyle(level: LogLevel): { bg: string; text: string; label: string } {
  switch (level) {
    case 'debug':
      return { bg: 'bg-gray-100', text: 'text-gray-500', label: 'DBG' };
    case 'info':
      return { bg: 'bg-blue-100', text: 'text-blue-700', label: 'INF' };
    case 'warn':
      return { bg: 'bg-yellow-100', text: 'text-yellow-700', label: 'WRN' };
    case 'error':
      return { bg: 'bg-red-100', text: 'text-red-700', label: 'ERR' };
  }
}

/**
 * Get row background color for a log level
 */
function getRowBg(level: LogLevel): string {
  switch (level) {
    case 'error':
      return 'bg-red-50';
    case 'warn':
      return 'bg-yellow-50';
    default:
      return '';
  }
}

/**
 * Individual log row
 */
function LogRow({ entry }: { entry: LogEntry }) {
  const style = getLevelStyle(entry.level);
  const rowBg = getRowBg(entry.level);

  return (
    <div className={`flex items-start gap-2 px-3 py-1 border-b border-gray-100 font-mono text-xs ${rowBg}`}>
      <span className="text-gray-400 flex-shrink-0 w-20">
        {formatTimestamp(entry.timestamp)}
      </span>
      <span className={`flex-shrink-0 px-1 rounded ${style.bg} ${style.text} font-semibold w-8 text-center`}>
        {style.label}
      </span>
      <span className="text-gray-500 flex-shrink-0 w-20 truncate" title={entry.source}>
        {entry.source}
      </span>
      <span className="text-gray-900 flex-1 break-words">
        {entry.message}
      </span>
    </div>
  );
}

type LevelFilter = 'all' | LogLevel;

/**
 * Level filter button
 */
function LevelButton({
  level,
  label,
  count,
  active,
  onClick,
}: {
  level: LevelFilter;
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  const getColor = () => {
    if (!active) return 'bg-gray-100 text-gray-600 hover:bg-gray-200';
    switch (level) {
      case 'error':
        return 'bg-red-100 text-red-700';
      case 'warn':
        return 'bg-yellow-100 text-yellow-700';
      case 'info':
        return 'bg-blue-100 text-blue-700';
      case 'debug':
        return 'bg-gray-200 text-gray-700';
      default:
        return 'bg-gray-200 text-gray-700';
    }
  };

  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 text-xs font-medium rounded transition-colors ${getColor()}`}
    >
      {label} ({count})
    </button>
  );
}

/**
 * LogsPanel component
 * Displays real-time pipeline log entries with filtering and auto-scroll
 */
export function LogsPanel() {
  const logs = useRuntimeStore((state) => state.logs);
  const clearLogs = useRuntimeStore((state) => state.clearLogs);

  const [levelFilter, setLevelFilter] = useState<LevelFilter>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [searchText, setSearchText] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);

  const listRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);

  // Collect unique sources for the dropdown
  const sources = Array.from(new Set(logs.map((l) => l.source))).sort();

  // Filter logs
  const filtered = logs.filter((entry) => {
    if (levelFilter !== 'all' && entry.level !== levelFilter) return false;
    if (sourceFilter !== 'all' && entry.source !== sourceFilter) return false;
    if (searchText) {
      const needle = searchText.toLowerCase();
      if (
        !entry.message.toLowerCase().includes(needle) &&
        !entry.source.toLowerCase().includes(needle)
      ) {
        return false;
      }
    }
    return true;
  });

  // Count by level (from full log set, not filtered)
  const counts = { all: logs.length, debug: 0, info: 0, warn: 0, error: 0 };
  for (const l of logs) {
    counts[l.level]++;
  }

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [filtered.length, autoScroll]);

  // Detect manual scroll to disable auto-scroll
  const handleScroll = useCallback(() => {
    if (!listRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 30;

    if (!atBottom && !isUserScrolling.current) {
      isUserScrolling.current = true;
      setAutoScroll(false);
    } else if (atBottom && isUserScrolling.current) {
      isUserScrolling.current = false;
      setAutoScroll(true);
    }
  }, []);

  // Copy visible logs to clipboard
  const handleCopy = useCallback(() => {
    const text = filtered
      .map((e) => `${formatTimestamp(e.timestamp)} [${e.level.toUpperCase()}] ${e.source}: ${e.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
  }, [filtered]);

  // Export logs as JSON
  const handleExport = useCallback(() => {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [filtered]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <LevelButton level="all" label="All" count={counts.all} active={levelFilter === 'all'} onClick={() => setLevelFilter('all')} />
            <LevelButton level="error" label="Error" count={counts.error} active={levelFilter === 'error'} onClick={() => setLevelFilter('error')} />
            <LevelButton level="warn" label="Warn" count={counts.warn} active={levelFilter === 'warn'} onClick={() => setLevelFilter('warn')} />
            <LevelButton level="info" label="Info" count={counts.info} active={levelFilter === 'info'} onClick={() => setLevelFilter('info')} />
            <LevelButton level="debug" label="Debug" count={counts.debug} active={levelFilter === 'debug'} onClick={() => setLevelFilter('debug')} />
          </div>

          {/* Source filter */}
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            className="text-xs border border-gray-300 rounded px-1 py-1 bg-white"
          >
            <option value="all">All sources</option>
            {sources.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>

          {/* Search */}
          <input
            type="text"
            placeholder="Search logs..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="text-xs border border-gray-300 rounded px-2 py-1 w-36 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center gap-2">
          {filtered.length !== logs.length && logs.length > 0 && (
            <span className="text-xs text-gray-500">
              {filtered.length} of {logs.length}
            </span>
          )}
          {!autoScroll && (
            <button
              onClick={() => {
                setAutoScroll(true);
                isUserScrolling.current = false;
                if (listRef.current) {
                  listRef.current.scrollTop = listRef.current.scrollHeight;
                }
              }}
              className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 flex items-center gap-1"
              title="Scroll to bottom"
            >
              <ArrowDownToLine className="w-3 h-3" />
              Auto-scroll
            </button>
          )}
          <button
            onClick={handleExport}
            className="px-2 py-1 text-xs bg-gray-500 text-white rounded hover:bg-gray-600 flex items-center gap-1"
            title="Export logs"
            disabled={filtered.length === 0}
          >
            <Download className="w-3 h-3" />
            Export
          </button>
          <button
            onClick={clearLogs}
            className="px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 flex items-center gap-1"
            title="Clear logs"
            disabled={logs.length === 0}
          >
            <Trash2 className="w-3 h-3" />
            Clear
          </button>
        </div>
      </div>

      {/* Log list */}
      <div
        ref={listRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto"
      >
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            {logs.length === 0
              ? 'No logs yet — start a pipeline to see log output'
              : 'No logs match the current filters'}
          </div>
        ) : (
          filtered.map((entry) => (
            <LogRow key={entry.id} entry={entry} />
          ))
        )}
      </div>
    </div>
  );
}
