import { useCallback, useState } from 'react';
import { FolderOpen } from 'lucide-react';
import type { PropertySchema } from '../../../types/schema';
import { FileBrowserDialog } from '../../FileBrowser';

interface PathInputProps {
  value: string;
  schema: PropertySchema;
  onChange: (value: string) => void;
  label: string;
  mode?: 'file' | 'directory' | 'both';
}

/**
 * Path input with browse button for file/directory selection
 */
export function PathInput({ value, schema, onChange, label, mode = 'both' }: PathInputProps) {
  const [browserOpen, setBrowserOpen] = useState(false);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  const handleBrowse = useCallback(() => {
    setBrowserOpen(true);
  }, []);

  const handleSelect = useCallback(
    (path: string) => {
      onChange(path);
    },
    [onChange]
  );

  const inputId = `prop-path-${label}`;

  // Determine mode from property name heuristics
  const inferredMode = label.toLowerCase().includes('dir') ? 'directory' : mode;

  // Determine filter from property name heuristics
  let filter: string | undefined;
  if (label.toLowerCase().includes('json')) filter = '.json';
  else if (label.toLowerCase().includes('image') || label.toLowerCase().includes('img')) filter = '.jpg';
  else if (label.toLowerCase().includes('model')) filter = '.onnx';

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      <div className="flex gap-1">
        <input
          id={inputId}
          type="text"
          value={value}
          onChange={handleChange}
          placeholder={schema.description || 'Enter path...'}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
        />
        <button
          type="button"
          onClick={handleBrowse}
          className="px-2 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 focus:outline-none focus:ring-1 focus:ring-blue-500"
          title="Browse..."
        >
          <FolderOpen className="w-4 h-4" />
        </button>
      </div>
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}

      <FileBrowserDialog
        isOpen={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={handleSelect}
        title={`Select ${inferredMode === 'directory' ? 'Directory' : 'File'}`}
        mode={inferredMode}
        filter={filter}
        initialPath={value || undefined}
      />
    </div>
  );
}
