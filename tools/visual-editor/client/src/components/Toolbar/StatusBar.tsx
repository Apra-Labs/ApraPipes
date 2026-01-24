import { useEffect, useState } from 'react';
import { Circle, AlertCircle, AlertTriangle, CheckCircle, Wifi, WifiOff, Cpu } from 'lucide-react';
import { useRuntimeStore } from '../../store/runtimeStore';
import { usePipelineStore } from '../../store/pipelineStore';
import { useWorkspaceStore } from '../../store/workspaceStore';
import { api } from '../../services/api';

/**
 * Format duration as HH:MM:SS
 */
function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Status bar component showing pipeline status and validation summary
 */
export function StatusBar() {
  // Runtime state
  const runtimeStatus = useRuntimeStore((state) => state.status);
  const connectionState = useRuntimeStore((state) => state.connectionState);
  const startTime = useRuntimeStore((state) => state.startTime);

  // Validation state
  const validationResult = usePipelineStore((state) => state.validationResult);

  // Workspace state
  const workspacePath = useWorkspaceStore((state) => state.currentPath);

  // Duration timer
  const [duration, setDuration] = useState<number>(0);

  // Addon status
  const [addonLoaded, setAddonLoaded] = useState<boolean | null>(null);
  const [moduleCount, setModuleCount] = useState<number>(0);

  useEffect(() => {
    if (runtimeStatus === 'RUNNING' && startTime) {
      const interval = setInterval(() => {
        setDuration(Date.now() - startTime);
      }, 1000);
      return () => clearInterval(interval);
    } else {
      setDuration(0);
    }
  }, [runtimeStatus, startTime]);

  // Fetch addon status on mount
  useEffect(() => {
    api.getSchemaStatus()
      .then((status) => {
        setAddonLoaded(status.addonLoaded);
        setModuleCount(status.moduleCount);
      })
      .catch(() => {
        setAddonLoaded(false);
      });
  }, []);

  // Count errors and warnings from validation
  const errorCount = validationResult?.issues.filter((i) => i.level === 'error').length ?? 0;
  const warningCount = validationResult?.issues.filter((i) => i.level === 'warning').length ?? 0;

  // Map runtime status to display status
  const displayStatus = runtimeStatus.toLowerCase() as 'idle' | 'running' | 'stopped' | 'error' | 'creating' | 'stopping';

  return (
    <footer className="h-6 border-t border-border bg-muted/50 flex items-center px-4 text-xs">
      {/* Pipeline Status */}
      <div className="flex items-center gap-2">
        <StatusIndicator status={displayStatus} />
        <span className="text-muted-foreground capitalize">{displayStatus}</span>
        {runtimeStatus === 'RUNNING' && startTime && (
          <span className="text-muted-foreground">
            ({formatDuration(duration)})
          </span>
        )}
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

      {/* Addon Status Indicator */}
      <div className="flex items-center gap-1 mr-3" title={addonLoaded ? `Native addon loaded (${moduleCount} modules)` : 'Mock mode - native addon not available'}>
        <Cpu className={`w-3 h-3 ${addonLoaded === null ? 'text-gray-400' : addonLoaded ? 'text-green-500' : 'text-yellow-500'}`} />
        <span className={`text-xs ${addonLoaded === null ? 'text-gray-400' : addonLoaded ? 'text-green-600' : 'text-yellow-600'}`}>
          {addonLoaded === null ? '...' : addonLoaded ? 'Native' : 'Mock'}
        </span>
      </div>

      {/* Divider */}
      <div className="w-px h-4 bg-border mr-3" />

      {/* WebSocket Connection Status */}
      <div className="flex items-center gap-1 mr-3">
        {connectionState === 'connected' ? (
          <Wifi className="w-3 h-3 text-green-500" />
        ) : connectionState === 'reconnecting' ? (
          <Wifi className="w-3 h-3 text-yellow-500 animate-pulse" />
        ) : (
          <WifiOff className="w-3 h-3 text-gray-400" />
        )}
      </div>

      {/* Workspace Path */}
      {workspacePath && (
        <span className="text-muted-foreground truncate max-w-xs">
          {workspacePath}
        </span>
      )}
    </footer>
  );
}

type DisplayStatus = 'idle' | 'running' | 'stopped' | 'error' | 'creating' | 'stopping';

function StatusIndicator({ status }: { status: DisplayStatus }) {
  const colors: Record<DisplayStatus, string> = {
    idle: 'text-gray-400',
    running: 'text-green-500 animate-pulse',
    stopped: 'text-gray-500',
    error: 'text-red-500',
    creating: 'text-yellow-500 animate-pulse',
    stopping: 'text-yellow-500 animate-pulse',
  };

  return <Circle className={`w-2 h-2 fill-current ${colors[status]}`} />;
}
