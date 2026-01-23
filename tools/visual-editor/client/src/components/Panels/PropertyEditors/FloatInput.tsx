import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface FloatInputProps {
  value: number;
  schema: PropertySchema;
  onChange: (value: number) => void;
  label: string;
}

/**
 * Float input with min/max validation and step control
 */
export function FloatInput({ value, schema, onChange, label }: FloatInputProps) {
  const min = schema.min !== undefined ? parseFloat(schema.min) : undefined;
  const max = schema.max !== undefined ? parseFloat(schema.max) : undefined;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(e.target.value);
      if (!isNaN(newValue)) {
        onChange(newValue);
      }
    },
    [onChange]
  );

  const inputId = `prop-float-${label}`;

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
        step={0.01}
        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
      />
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
