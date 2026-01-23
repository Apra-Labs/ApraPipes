import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface BoolCheckboxProps {
  value: boolean;
  schema: PropertySchema;
  onChange: (value: boolean) => void;
  label: string;
}

/**
 * Boolean checkbox editor
 */
export function BoolCheckbox({ value, schema, onChange, label }: BoolCheckboxProps) {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.checked);
    },
    [onChange]
  );

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={value}
          onChange={handleChange}
          id={`prop-${label}`}
          className="w-4 h-4 rounded border-gray-300 text-blue-500 focus:ring-blue-500"
        />
        <label htmlFor={`prop-${label}`} className="text-xs font-medium text-gray-600">
          {label}
        </label>
      </div>
      {schema.description && (
        <p className="text-xs text-gray-400 ml-6">{schema.description}</p>
      )}
    </div>
  );
}
