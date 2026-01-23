/**
 * File Browser Dialog
 *
 * A modal dialog for browsing the server's file system.
 * Used for file/path property selection and workspace open/save dialogs.
 */

import { useState, useCallback, useEffect } from 'react';
import { Folder, File, ChevronRight, Home, X, Loader2 } from 'lucide-react';

interface FileEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
}

interface FileBrowserDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  title?: string;
  mode?: 'file' | 'directory' | 'both';
  filter?: string; // File extension filter, e.g., ".json"
  initialPath?: string;
}

export function FileBrowserDialog({
  isOpen,
  onClose,
  onSelect,
  title = 'Select File',
  mode = 'file',
  filter,
  initialPath,
}: FileBrowserDialogProps) {
  const [currentPath, setCurrentPath] = useState<string>('');
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<FileEntry | null>(null);
  const [pathInput, setPathInput] = useState('');

  // Load directory contents
  const loadDirectory = useCallback(async (path?: string) => {
    setLoading(true);
    setError(null);
    setSelectedEntry(null);

    try {
      const params = new URLSearchParams();
      if (path) params.set('path', path);
      if (filter && mode === 'file') params.set('filter', filter);

      const url = `/api/files/list?${params}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setCurrentPath(data.path);
      setParentPath(data.parent);
      setEntries(data.entries);
      setPathInput(data.path);
    } catch (err) {
      console.error('FileBrowser error:', err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Cannot connect to server. Is it running?');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load directory');
      }
    } finally {
      setLoading(false);
    }
  }, [filter, mode]);

  // Load home directory on open
  useEffect(() => {
    if (isOpen) {
      loadDirectory(initialPath || undefined);
    }
  }, [isOpen, initialPath, loadDirectory]);

  const handleEntryClick = useCallback((entry: FileEntry) => {
    if (entry.type === 'directory') {
      loadDirectory(entry.path);
    } else {
      setSelectedEntry(entry);
    }
  }, [loadDirectory]);

  const handleEntryDoubleClick = useCallback((entry: FileEntry) => {
    if (entry.type === 'directory') {
      if (mode === 'directory') {
        onSelect(entry.path);
        onClose();
      } else {
        loadDirectory(entry.path);
      }
    } else {
      onSelect(entry.path);
      onClose();
    }
  }, [loadDirectory, mode, onSelect, onClose]);

  const handleGoUp = useCallback(() => {
    if (parentPath) {
      loadDirectory(parentPath);
    }
  }, [parentPath, loadDirectory]);

  const handleGoHome = useCallback(async () => {
    try {
      const response = await fetch('/api/files/home');
      const data = await response.json();
      loadDirectory(data.path);
    } catch {
      loadDirectory(undefined);
    }
  }, [loadDirectory]);

  const handlePathSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    loadDirectory(pathInput);
  }, [pathInput, loadDirectory]);

  const handleSelect = useCallback(() => {
    if (mode === 'directory') {
      onSelect(currentPath);
    } else if (selectedEntry) {
      onSelect(selectedEntry.path);
    }
    onClose();
  }, [mode, currentPath, selectedEntry, onSelect, onClose]);

  if (!isOpen) return null;

  const canSelect = mode === 'directory' || (mode === 'file' && selectedEntry?.type === 'file') || (mode === 'both' && selectedEntry);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg shadow-xl w-[600px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold">{title}</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-200">
          <button
            onClick={handleGoHome}
            className="p-1.5 hover:bg-gray-100 rounded"
            title="Go to home directory"
          >
            <Home className="w-4 h-4" />
          </button>
          <button
            onClick={handleGoUp}
            disabled={!parentPath}
            className="p-1.5 hover:bg-gray-100 rounded disabled:opacity-50"
            title="Go up"
          >
            <ChevronRight className="w-4 h-4 rotate-180" />
          </button>
          <form onSubmit={handlePathSubmit} className="flex-1">
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
            />
          </form>
        </div>

        {/* File list */}
        <div className="flex-1 overflow-y-auto min-h-[300px]">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full text-red-500">
              {error}
            </div>
          ) : entries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              Empty directory
            </div>
          ) : (
            <div className="divide-y divide-gray-100">
              {entries.map((entry) => (
                <div
                  key={entry.path}
                  onClick={() => handleEntryClick(entry)}
                  onDoubleClick={() => handleEntryDoubleClick(entry)}
                  className={`
                    flex items-center gap-3 px-4 py-2 cursor-pointer hover:bg-gray-50
                    ${selectedEntry?.path === entry.path ? 'bg-blue-50' : ''}
                  `}
                >
                  {entry.type === 'directory' ? (
                    <Folder className="w-5 h-5 text-yellow-500" />
                  ) : (
                    <File className="w-5 h-5 text-gray-400" />
                  )}
                  <span className="flex-1 truncate">{entry.name}</span>
                  {entry.type === 'file' && entry.size !== undefined && (
                    <span className="text-xs text-gray-400">
                      {formatSize(entry.size)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-500">
            {mode === 'directory' ? (
              <span>Current: {currentPath}</span>
            ) : selectedEntry ? (
              <span>Selected: {selectedEntry.name}</span>
            ) : (
              <span>Select a {mode === 'both' ? 'file or directory' : mode}</span>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!canSelect}
              className="px-4 py-2 text-sm text-white bg-blue-500 hover:bg-blue-600 rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
