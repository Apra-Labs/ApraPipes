import { useCallback } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface FloatInputProps {
  value: number;
  schema: PropertySchema;
  onChange: (value: number) => void;
  label: string;
}

/**
 * Float input with slider when min/max are defined, otherwise number input
 */
export function FloatInput({ value, schema, onChange, label }: FloatInputProps) {
  const min = schema.min !== undefined ? parseFloat(schema.min) : undefined;
  const max = schema.max !== undefined ? parseFloat(schema.max) : undefined;
  const hasRange = min !== undefined && max !== undefined && !isNaN(min) && !isNaN(max);

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
  const sliderId = `prop-float-slider-${label}`;

  // Calculate step for slider based on range
  const step = hasRange ? (max - min) / 100 : 0.01;

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      {hasRange ? (
        // Slider with number input for precise control
        <div className="flex items-center gap-2">
          <input
            id={sliderId}
            type="range"
            value={value}
            onChange={handleChange}
            min={min}
            max={max}
            step={step}
            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <input
            id={inputId}
            type="number"
            value={value}
            onChange={handleChange}
            min={min}
            max={max}
            step={step}
            className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      ) : (
        // Standard number input when no range defined
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
      )}
      {hasRange && (
        <div className="flex justify-between text-xs text-gray-400">
          <span>{min}</span>
          <span>{max}</span>
        </div>
      )}
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
