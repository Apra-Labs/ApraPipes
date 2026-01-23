import { useCallback, useState } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface JsonEditorProps {
  value: unknown;
  schema: PropertySchema;
  onChange: (value: unknown) => void;
  label: string;
}

/**
 * JSON editor using a textarea (simple version for v1)
 */
export function JsonEditor({ value, schema, onChange, label }: JsonEditorProps) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newText = e.target.value;
      setText(newText);

      try {
        const parsed = JSON.parse(newText);
        onChange(parsed);
        setError(null);
      } catch {
        setError('Invalid JSON');
      }
    },
    [onChange]
  );

  return (
    <div className="space-y-1">
      <label className="block text-xs font-medium text-gray-600">{label}</label>
      <textarea
        value={text}
        onChange={handleChange}
        rows={4}
        className={`w-full px-2 py-1 text-sm font-mono border rounded focus:outline-none focus:ring-1 ${
          error
            ? 'border-red-500 focus:ring-red-500'
            : 'border-gray-300 focus:ring-blue-500'
        }`}
      />
      {error && <p className="text-xs text-red-500">{error}</p>}
      {schema.description && !error && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
