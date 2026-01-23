import { useCallback, useMemo } from 'react';
import type { PropertySchema } from '../../../types/schema';

interface FloatInputProps {
  value: number;
  schema: PropertySchema;
  onChange: (value: number) => void;
  label: string;
}

/**
 * Float input with slider + number input for precise control
 * Always shows slider with sensible defaults when min/max not defined
 */
export function FloatInput({ value, schema, onChange, label }: FloatInputProps) {
  // Parse min/max from schema, with smart defaults
  const { sliderMin, sliderMax, step } = useMemo(() => {
    const schemaMin = schema.min !== undefined ? parseFloat(schema.min) : undefined;
    const schemaMax = schema.max !== undefined ? parseFloat(schema.max) : undefined;

    // If both min and max are defined, use them
    if (schemaMin !== undefined && schemaMax !== undefined && !isNaN(schemaMin) && !isNaN(schemaMax)) {
      const range = schemaMax - schemaMin;
      return {
        sliderMin: schemaMin,
        sliderMax: schemaMax,
        step: range > 0 ? range / 100 : 0.01,
      };
    }

    // Smart defaults based on value and common patterns
    // For normalized values (0-1 range)
    if (value >= 0 && value <= 1) {
      return { sliderMin: 0, sliderMax: 1, step: 0.01 };
    }
    // For percentage-like values
    if (value >= 0 && value <= 100) {
      return { sliderMin: 0, sliderMax: 100, step: 1 };
    }
    // For other values, create a range around the current value
    const absVal = Math.abs(value) || 1;
    const magnitude = Math.pow(10, Math.floor(Math.log10(absVal)));
    return {
      sliderMin: schemaMin ?? 0,
      sliderMax: schemaMax ?? Math.max(absVal * 2, magnitude * 10),
      step: magnitude / 100,
    };
  }, [schema.min, schema.max, value]);

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

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="block text-xs font-medium text-gray-600">{label}</label>
      {/* Always show slider with number input for precise control */}
      <div className="flex items-center gap-2">
        <input
          id={sliderId}
          type="range"
          value={value}
          onChange={handleChange}
          min={sliderMin}
          max={sliderMax}
          step={step}
          className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <input
          id={inputId}
          type="number"
          value={value}
          onChange={handleChange}
          step={step}
          className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400">
        <span>{sliderMin}</span>
        <span>{sliderMax}</span>
      </div>
      {schema.description && (
        <p className="text-xs text-gray-400">{schema.description}</p>
      )}
    </div>
  );
}
