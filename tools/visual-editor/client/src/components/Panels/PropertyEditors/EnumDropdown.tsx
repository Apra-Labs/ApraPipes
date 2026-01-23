import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface EnumDropdownProps {
  value: string;
  schema: PropertySchema;
  onChange: (value: string) => void;
  label: string;
}

/**
 * Enum dropdown selector
 */
export function EnumDropdown({ value, schema, onChange, label }: EnumDropdownProps) {
  const options = schema.enum_values ?? [];

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange(e.target.value);
    },
    [onChange]
  );

  const inputId = `prop-enum-${label}`;

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      <select
        id={inputId}
        value={value}
        onChange={handleChange}
        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
