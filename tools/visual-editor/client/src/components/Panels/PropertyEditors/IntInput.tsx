import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface IntInputProps {
  value: number;
  schema: PropertySchema;
  onChange: (value: number) => void;
  label: string;
}

/**
 * Integer input with min/max validation from schema
 */
export function IntInput({ value, schema, onChange, label }: IntInputProps) {
  const min = schema.min !== undefined ? parseInt(schema.min, 10) : undefined;
  const max = schema.max !== undefined ? parseInt(schema.max, 10) : undefined;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseInt(e.target.value, 10);
      if (!isNaN(newValue)) {
        onChange(newValue);
      }
    },
    [onChange]
  );

  const inputId = `prop-int-${label}`;

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      <input
        id={inputId}
        type="number"
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        step={1}
        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
      />
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
