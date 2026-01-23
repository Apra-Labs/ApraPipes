import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface StringInputProps {
  value: string;
  schema: PropertySchema;
  onChange: (value: string) => void;
  label: string;
}

/**
 * String text input editor
 */
export function StringInput({ value, schema, onChange, label }: StringInputProps) {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  const inputId = `prop-string-${label}`;

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      <input
        id={inputId}
        type="text"
        value={value}
        onChange={handleChange}
        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
        placeholder={schema.default ?? ''}
      />
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
