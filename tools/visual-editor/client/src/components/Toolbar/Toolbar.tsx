import { useCallback, useState, useEffect } from 'react';
import { Play, Square, Save, FolderOpen, FileJson, Settings, FilePlus, Loader2 } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import { useWorkspaceStore } from '../../store/workspaceStore';
import { useRuntimeStore } from '../../store/runtimeStore';
import { usePipelineStore } from '../../store/pipelineStore';

/**
 * Main toolbar component with pipeline controls
 */
export function Toolbar() {
  const viewMode = useUIStore((state) => state.viewMode);
  const setViewMode = useUIStore((state) => state.setViewMode);

  const currentPath = useWorkspaceStore((state) => state.currentPath);
  const isDirty = useWorkspaceStore((state) => state.isDirty);
  const newWorkspace = useWorkspaceStore((state) => state.newWorkspace);
  const saveWorkspace = useWorkspaceStore((state) => state.saveWorkspace);
  const openWorkspace = useWorkspaceStore((state) => state.openWorkspace);

  // Runtime state
  const runtimeStatus = useRuntimeStore((state) => state.status);
  const isLoading = useRuntimeStore((state) => state.isLoading);
  const createPipeline = useRuntimeStore((state) => state.createPipeline);
  const startPipeline = useRuntimeStore((state) => state.startPipeline);
  const stopPipeline = useRuntimeStore((state) => state.stopPipeline);
  const deletePipeline = useRuntimeStore((state) => state.deletePipeline);
  const connect = useRuntimeStore((state) => state.connect);
  const pipelineId = useRuntimeStore((state) => state.pipelineId);

  // Pipeline config
  const pipelineConfig = usePipelineStore((state) => state.config);
  const validatePipeline = usePipelineStore((state) => state.validate);

  const [openDialogVisible, setOpenDialogVisible] = useState(false);
  const [saveDialogVisible, setSaveDialogVisible] = useState(false);
  const [pathInput, setPathInput] = useState('');

  // Connect to WebSocket on mount
  useEffect(() => {
    connect();
  }, [connect]);

  const handleNew = useCallback(() => {
    if (isDirty) {
      const confirmed = window.confirm('You have unsaved changes. Create new workspace anyway?');
      if (!confirmed) return;
    }
    newWorkspace();
  }, [isDirty, newWorkspace]);

  const handleOpen = useCallback(() => {
    setPathInput('');
    setOpenDialogVisible(true);
  }, []);

  const handleOpenConfirm = useCallback(async () => {
    if (!pathInput.trim()) return;
    try {
      await openWorkspace(pathInput.trim());
      setOpenDialogVisible(false);
    } catch (error) {
      alert(`Failed to open workspace: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [pathInput, openWorkspace]);

  const handleSave = useCallback(async () => {
    if (currentPath) {
      try {
        await saveWorkspace();
      } catch (error) {
        alert(`Failed to save workspace: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    } else {
      setPathInput('');
      setSaveDialogVisible(true);
    }
  }, [currentPath, saveWorkspace]);

  const handleSaveConfirm = useCallback(async () => {
    if (!pathInput.trim()) return;
    try {
      await saveWorkspace(pathInput.trim());
      setSaveDialogVisible(false);
    } catch (error) {
      alert(`Failed to save workspace: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [pathInput, saveWorkspace]);

  // Run pipeline
  const handleRun = useCallback(async () => {
    try {
      // Create pipeline if we don't have one
      if (!pipelineId) {
        await createPipeline(pipelineConfig);
      }
      // Start the pipeline
      await startPipeline();
    } catch (error) {
      alert(`Failed to run pipeline: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [pipelineId, pipelineConfig, createPipeline, startPipeline]);

  // Stop pipeline
  const handleStop = useCallback(async () => {
    try {
      await stopPipeline();
      // Optionally delete after stopping
      await deletePipeline();
    } catch (error) {
      alert(`Failed to stop pipeline: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [stopPipeline, deletePipeline]);

  // Validate pipeline
  const handleValidate = useCallback(async () => {
    try {
      const result = await validatePipeline();
      if (result.valid) {
        alert('Pipeline is valid!');
      } else {
        const errorCount = result.issues.filter((i) => i.level === 'error').length;
        const warningCount = result.issues.filter((i) => i.level === 'warning').length;
        alert(`Validation complete: ${errorCount} error(s), ${warningCount} warning(s)`);
      }
    } catch (error) {
      alert(`Validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [validatePipeline]);

  const canRun = runtimeStatus === 'IDLE' || runtimeStatus === 'STOPPED';
  const canStop = runtimeStatus === 'RUNNING';
  const hasModules = Object.keys(pipelineConfig.modules).length > 0;

  return (
    <>
      <header className="h-12 border-b border-border bg-background flex items-center px-4 gap-2">
        {/* Logo/Title */}
        <div className="flex items-center gap-2 mr-4">
          <span className="font-bold text-lg">ApraPipes Studio</span>
        </div>

        {/* Divider */}
        <div className="w-px h-6 bg-border" />

        {/* File Operations */}
        <div className="flex items-center gap-1 ml-2">
          <ToolbarButton icon={<FilePlus className="w-4 h-4" />} label="New" onClick={handleNew} />
          <ToolbarButton icon={<FolderOpen className="w-4 h-4" />} label="Open" onClick={handleOpen} />
          <ToolbarButton
            icon={<Save className="w-4 h-4" />}
            label={isDirty ? 'Save*' : 'Save'}
            onClick={handleSave}
          />
        </div>

        {/* Divider */}
        <div className="w-px h-6 bg-border" />

        {/* Pipeline Controls */}
        <div className="flex items-center gap-1 ml-2">
          <ToolbarButton
            icon={isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            label={runtimeStatus === 'RUNNING' ? 'Running...' : 'Run'}
            variant="success"
            disabled={!canRun || !hasModules || isLoading}
            onClick={handleRun}
          />
          <ToolbarButton
            icon={<Square className="w-4 h-4" />}
            label="Stop"
            variant="danger"
            disabled={!canStop || isLoading}
            onClick={handleStop}
          />
        </div>

        {/* Divider */}
        <div className="w-px h-6 bg-border" />

        {/* View Controls */}
        <div className="flex items-center gap-1 ml-2">
          <ViewToggleButton active={viewMode === 'visual'} onClick={() => setViewMode('visual')}>
            Visual
          </ViewToggleButton>
          <ViewToggleButton active={viewMode === 'json'} onClick={() => setViewMode('json')}>
            JSON
          </ViewToggleButton>
          <ViewToggleButton active={viewMode === 'split'} onClick={() => setViewMode('split')}>
            Split
          </ViewToggleButton>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Current file indicator */}
        {currentPath && (
          <span className="text-sm text-muted-foreground mr-2">
            {currentPath}
            {isDirty && <span className="text-yellow-500"> *</span>}
          </span>
        )}

        {/* Right side */}
        <div className="flex items-center gap-1">
          <ToolbarButton
            icon={<FileJson className="w-4 h-4" />}
            label="Validate"
            disabled={!hasModules}
            onClick={handleValidate}
          />
          <ToolbarButton icon={<Settings className="w-4 h-4" />} label="Settings" disabled />
        </div>
      </header>

      {/* Open Dialog */}
      {openDialogVisible && (
        <Dialog
          title="Open Workspace"
          onClose={() => setOpenDialogVisible(false)}
          onConfirm={handleOpenConfirm}
        >
          <input
            type="text"
            placeholder="Enter workspace path (e.g., my-project)"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoFocus
          />
        </Dialog>
      )}

      {/* Save Dialog */}
      {saveDialogVisible && (
        <Dialog
          title="Save Workspace"
          onClose={() => setSaveDialogVisible(false)}
          onConfirm={handleSaveConfirm}
        >
          <input
            type="text"
            placeholder="Enter workspace path (e.g., my-project)"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoFocus
          />
        </Dialog>
      )}
    </>
  );
}

interface ToolbarButtonProps {
  icon: React.ReactNode;
  label: string;
  variant?: 'default' | 'success' | 'danger';
  disabled?: boolean;
  onClick?: () => void;
}

function ToolbarButton({
  icon,
  label,
  variant = 'default',
  disabled = false,
  onClick,
}: ToolbarButtonProps) {
  const variantClasses = {
    default: 'hover:bg-muted',
    success: 'hover:bg-green-100 text-green-700',
    danger: 'hover:bg-red-100 text-red-700',
  };

  return (
    <button
      className={`
        flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium
        transition-colors
        ${variantClasses[variant]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
      disabled={disabled}
      onClick={onClick}
      title={label}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

interface ViewToggleButtonProps {
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
}

function ViewToggleButton({ children, active, onClick }: ViewToggleButtonProps) {
  return (
    <button
      className={`
        px-3 py-1 text-sm font-medium rounded transition-colors
        ${active ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}
      `}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

interface DialogProps {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
  onConfirm: () => void;
}

function Dialog({ title, children, onClose, onConfirm }: DialogProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg shadow-xl w-96 p-4">
        <h2 className="text-lg font-semibold mb-4">{title}</h2>
        <div className="mb-4">{children}</div>
        <div className="flex justify-end gap-2">
          <button
            className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="px-4 py-2 text-sm text-white bg-blue-500 hover:bg-blue-600 rounded"
            onClick={onConfirm}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
