import { Circle, AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

type PipelineStatus = 'idle' | 'running' | 'stopped' | 'error';

interface StatusBarProps {
  status?: PipelineStatus;
  errorCount?: number;
  warningCount?: number;
  workspacePath?: string;
}

/**
 * Status bar component showing pipeline status and validation summary
 */
export function StatusBar({
  status = 'idle',
  errorCount = 0,
  warningCount = 0,
  workspacePath,
}: StatusBarProps) {
  return (
    <footer className="h-6 border-t border-border bg-muted/50 flex items-center px-4 text-xs">
      {/* Pipeline Status */}
      <div className="flex items-center gap-2">
        <StatusIndicator status={status} />
        <span className="text-muted-foreground capitalize">{status}</span>
      </div>

      {/* Divider */}
      <div className="w-px h-4 bg-border mx-3" />

      {/* Validation Summary */}
      <div className="flex items-center gap-3">
        {errorCount > 0 ? (
          <span className="flex items-center gap-1 text-destructive">
            <AlertCircle className="w-3 h-3" />
            {errorCount} {errorCount === 1 ? 'error' : 'errors'}
          </span>
        ) : (
          <span className="flex items-center gap-1 text-green-600">
            <CheckCircle className="w-3 h-3" />
            No errors
          </span>
        )}
        {warningCount > 0 && (
          <span className="flex items-center gap-1 text-yellow-600">
            <AlertTriangle className="w-3 h-3" />
            {warningCount} {warningCount === 1 ? 'warning' : 'warnings'}
          </span>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Workspace Path */}
      {workspacePath && (
        <span className="text-muted-foreground truncate max-w-xs">
          {workspacePath}
        </span>
      )}
    </footer>
  );
}

function StatusIndicator({ status }: { status: PipelineStatus }) {
  const colors = {
    idle: 'text-gray-400',
    running: 'text-green-500 animate-pulse',
    stopped: 'text-gray-500',
    error: 'text-red-500',
  };

  return <Circle className={`w-2 h-2 fill-current ${colors[status]}`} />;
}
