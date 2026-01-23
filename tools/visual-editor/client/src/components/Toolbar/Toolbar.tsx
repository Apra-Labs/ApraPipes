import { useCallback, useState, useEffect, useRef } from 'react';
import { Play, Square, Save, FolderOpen, FileJson, Settings, FilePlus, Loader2, Undo, Redo, HelpCircle, X, Upload, Clock, ChevronDown, Trash2 } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import { useWorkspaceStore } from '../../store/workspaceStore';
import { useRuntimeStore } from '../../store/runtimeStore';
import { usePipelineStore } from '../../store/pipelineStore';
import { useCanvasStore } from '../../store/canvasStore';

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
  const recentFiles = useWorkspaceStore((state) => state.recentFiles);
  const clearRecentFiles = useWorkspaceStore((state) => state.clearRecentFiles);
  const importJSON = useWorkspaceStore((state) => state.importJSON);

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

  // Undo/Redo state
  const canUndo = useCanvasStore((state) => state.canUndo);
  const canRedo = useCanvasStore((state) => state.canRedo);
  const undo = useCanvasStore((state) => state.undo);
  const redo = useCanvasStore((state) => state.redo);

  const [openDialogVisible, setOpenDialogVisible] = useState(false);
  const [saveDialogVisible, setSaveDialogVisible] = useState(false);
  const [helpDialogVisible, setHelpDialogVisible] = useState(false);
  const [importDialogVisible, setImportDialogVisible] = useState(false);
  const [recentFilesVisible, setRecentFilesVisible] = useState(false);
  const [pathInput, setPathInput] = useState('');
  const [importContent, setImportContent] = useState('');
  const recentFilesRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Connect to WebSocket on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // Click outside handler for recent files dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (recentFilesRef.current && !recentFilesRef.current.contains(e.target as Node)) {
        setRecentFilesVisible(false);
      }
    };
    if (recentFilesVisible) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [recentFilesVisible]);

  // Get selected node for delete shortcut
  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const removeNode = useCanvasStore((state) => state.removeNode);

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

  // Import JSON from text
  const handleImport = useCallback(() => {
    setImportContent('');
    setImportDialogVisible(true);
  }, []);

  const handleImportConfirm = useCallback(() => {
    if (!importContent.trim()) return;
    try {
      importJSON(importContent.trim());
      setImportDialogVisible(false);
    } catch (error) {
      alert(`Failed to import JSON: ${error instanceof Error ? error.message : 'Invalid JSON'}`);
    }
  }, [importContent, importJSON]);

  // Import JSON from file
  const handleFileImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      try {
        importJSON(content);
      } catch (error) {
        alert(`Failed to import JSON: ${error instanceof Error ? error.message : 'Invalid JSON'}`);
      }
    };
    reader.readAsText(file);
    // Reset input so same file can be selected again
    e.target.value = '';
  }, [importJSON]);

  // Open recent file
  const handleOpenRecent = useCallback(async (path: string) => {
    setRecentFilesVisible(false);
    try {
      await openWorkspace(path);
    } catch (error) {
      alert(`Failed to open workspace: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [openWorkspace]);

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

  // Keyboard shortcuts (defined after all handlers)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle shortcuts when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Ctrl/Cmd + Z = Undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (canUndo) undo();
        return;
      }

      // Ctrl/Cmd + Shift + Z = Redo (or Ctrl/Cmd + Y)
      if ((e.ctrlKey || e.metaKey) && ((e.key === 'z' && e.shiftKey) || e.key === 'y')) {
        e.preventDefault();
        if (canRedo) redo();
        return;
      }

      // Ctrl/Cmd + S = Save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
        return;
      }

      // Ctrl/Cmd + N = New
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        handleNew();
        return;
      }

      // Ctrl/Cmd + O = Open
      if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        handleOpen();
        return;
      }

      // Delete or Backspace = Delete selected node
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNodeId) {
        e.preventDefault();
        removeNode(selectedNodeId);
        return;
      }

      // F5 = Validate
      if (e.key === 'F5') {
        e.preventDefault();
        handleValidate();
        return;
      }

      // ? = Show help (Shift + /)
      if (e.key === '?' || (e.shiftKey && e.key === '/')) {
        e.preventDefault();
        setHelpDialogVisible(true);
        return;
      }

      // Ctrl/Cmd + I = Import
      if ((e.ctrlKey || e.metaKey) && e.key === 'i') {
        e.preventDefault();
        handleImport();
        return;
      }

      // Escape = Close dialogs
      if (e.key === 'Escape') {
        setHelpDialogVisible(false);
        setOpenDialogVisible(false);
        setSaveDialogVisible(false);
        setImportDialogVisible(false);
        setRecentFilesVisible(false);
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [canUndo, canRedo, undo, redo, handleSave, handleNew, handleOpen, handleImport, selectedNodeId, removeNode, handleValidate]);

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
          <ToolbarButton icon={<Upload className="w-4 h-4" />} label="Import" onClick={handleImport} />

          {/* Recent Files Dropdown */}
          <div className="relative" ref={recentFilesRef}>
            <button
              className={`
                flex items-center gap-1 px-2 py-1.5 rounded text-sm font-medium
                transition-colors hover:bg-muted
                ${recentFiles.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}
              `}
              onClick={() => setRecentFilesVisible(!recentFilesVisible)}
              disabled={recentFiles.length === 0}
              title="Recent Files"
            >
              <Clock className="w-4 h-4" />
              <ChevronDown className="w-3 h-3" />
            </button>
            {recentFilesVisible && recentFiles.length > 0 && (
              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 w-64 max-h-80 overflow-y-auto">
                <div className="px-3 py-2 border-b border-gray-200 flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Recent Files</span>
                  <button
                    className="p-1 hover:bg-gray-100 rounded text-gray-500"
                    onClick={(e) => {
                      e.stopPropagation();
                      clearRecentFiles();
                      setRecentFilesVisible(false);
                    }}
                    title="Clear recent files"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
                <div className="py-1">
                  {recentFiles.map((file) => (
                    <button
                      key={file}
                      className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 truncate"
                      onClick={() => handleOpenRecent(file)}
                      title={file}
                    >
                      {file}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Hidden file input for importing JSON files */}
        <input
          type="file"
          ref={fileInputRef}
          accept=".json"
          onChange={handleFileImport}
          className="hidden"
        />

        {/* Divider */}
        <div className="w-px h-6 bg-border" />

        {/* Undo/Redo */}
        <div className="flex items-center gap-1 ml-2">
          <ToolbarButton
            icon={<Undo className="w-4 h-4" />}
            label="Undo"
            disabled={!canUndo}
            onClick={undo}
          />
          <ToolbarButton
            icon={<Redo className="w-4 h-4" />}
            label="Redo"
            disabled={!canRedo}
            onClick={redo}
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
          <ToolbarButton
            icon={<HelpCircle className="w-4 h-4" />}
            label="Help"
            onClick={() => setHelpDialogVisible(true)}
          />
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

      {/* Import Dialog */}
      {importDialogVisible && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-lg shadow-xl w-[500px] p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Import Pipeline JSON</h2>
              <button
                className="p-1 hover:bg-gray-100 rounded"
                onClick={() => fileInputRef.current?.click()}
                title="Import from file"
              >
                <FolderOpen className="w-5 h-5" />
              </button>
            </div>
            <div className="mb-4">
              <textarea
                placeholder="Paste pipeline JSON here..."
                value={importContent}
                onChange={(e) => setImportContent(e.target.value)}
                className="w-full h-48 px-3 py-2 border border-gray-300 rounded font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                autoFocus
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded"
                onClick={() => setImportDialogVisible(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm text-white bg-blue-500 hover:bg-blue-600 rounded disabled:opacity-50"
                onClick={handleImportConfirm}
                disabled={!importContent.trim()}
              >
                Import
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Help Dialog */}
      {helpDialogVisible && (
        <HelpModal onClose={() => setHelpDialogVisible(false)} />
      )}
    </>
  );
}

/**
 * Help modal showing keyboard shortcuts
 */
function HelpModal({ onClose }: { onClose: () => void }) {
  const shortcuts = [
    { key: 'Ctrl+N', action: 'New workspace' },
    { key: 'Ctrl+O', action: 'Open workspace' },
    { key: 'Ctrl+S', action: 'Save workspace' },
    { key: 'Ctrl+I', action: 'Import JSON' },
    { key: 'Ctrl+Z', action: 'Undo' },
    { key: 'Ctrl+Shift+Z', action: 'Redo' },
    { key: 'Ctrl+Y', action: 'Redo (alternative)' },
    { key: 'Delete', action: 'Delete selected node' },
    { key: 'Backspace', action: 'Delete selected node' },
    { key: 'F5', action: 'Validate pipeline' },
    { key: '?', action: 'Show this help' },
    { key: 'Escape', action: 'Close dialogs' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg shadow-xl w-96 max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-4 overflow-y-auto max-h-[60vh]">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-gray-500 border-b">
                <th className="pb-2">Shortcut</th>
                <th className="pb-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {shortcuts.map(({ key, action }) => (
                <tr key={key} className="border-b border-gray-100 last:border-0">
                  <td className="py-2">
                    <kbd className="px-2 py-1 text-sm bg-gray-100 border border-gray-300 rounded font-mono">
                      {key}
                    </kbd>
                  </td>
                  <td className="py-2 text-sm text-gray-700">{action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
          <p className="text-xs text-gray-500 text-center">
            Press <kbd className="px-1 bg-gray-100 border rounded">?</kbd> anytime to show this help
          </p>
        </div>
      </div>
    </div>
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
