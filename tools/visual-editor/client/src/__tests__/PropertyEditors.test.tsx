import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import {
  IntInput,
  FloatInput,
  BoolCheckbox,
  StringInput,
  EnumDropdown,
  JsonEditor,
} from '../components/Panels/PropertyEditors';
import type { PropertySchema } from '../types/schema';

describe('PropertyEditors', () => {
  describe('IntInput', () => {
    const schema: PropertySchema = {
      type: 'int',
      min: '0',
      max: '100',
      description: 'A test integer',
    };

    it('renders with label and value', () => {
      render(<IntInput label="Width" value={640} schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Width')).toHaveValue(640);
    });

    it('calls onChange with parsed integer', () => {
      const onChange = vi.fn();
      render(<IntInput label="Width" value={640} schema={schema} onChange={onChange} />);

      const input = screen.getByLabelText('Width');
      fireEvent.change(input, { target: { value: '800' } });

      expect(onChange).toHaveBeenCalledWith(800);
    });

    it('shows description', () => {
      render(<IntInput label="Width" value={640} schema={schema} onChange={() => {}} />);
      expect(screen.getByText('A test integer')).toBeInTheDocument();
    });

    it('respects min/max attributes', () => {
      render(<IntInput label="Width" value={50} schema={schema} onChange={() => {}} />);
      const input = screen.getByLabelText('Width');
      expect(input).toHaveAttribute('min', '0');
      expect(input).toHaveAttribute('max', '100');
    });
  });

  describe('FloatInput', () => {
    const schema: PropertySchema = {
      type: 'float',
      min: '0.0',
      max: '1.0',
    };

    it('renders with label and value', () => {
      render(<FloatInput label="Opacity" value={0.5} schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Opacity')).toHaveValue(0.5);
    });

    it('calls onChange with parsed float', () => {
      const onChange = vi.fn();
      render(<FloatInput label="Opacity" value={0.5} schema={schema} onChange={onChange} />);

      const input = screen.getByLabelText('Opacity');
      fireEvent.change(input, { target: { value: '0.75' } });

      expect(onChange).toHaveBeenCalledWith(0.75);
    });
  });

  describe('BoolCheckbox', () => {
    const schema: PropertySchema = {
      type: 'bool',
      description: 'Enable feature',
    };

    it('renders checked when value is true', () => {
      render(<BoolCheckbox label="Enabled" value={true} schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Enabled')).toBeChecked();
    });

    it('renders unchecked when value is false', () => {
      render(<BoolCheckbox label="Enabled" value={false} schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Enabled')).not.toBeChecked();
    });

    it('calls onChange with toggled value', () => {
      const onChange = vi.fn();
      render(<BoolCheckbox label="Enabled" value={false} schema={schema} onChange={onChange} />);

      fireEvent.click(screen.getByLabelText('Enabled'));

      expect(onChange).toHaveBeenCalledWith(true);
    });
  });

  describe('StringInput', () => {
    const schema: PropertySchema = {
      type: 'string',
      default: 'default.jpg',
      description: 'File path',
    };

    it('renders with label and value', () => {
      render(<StringInput label="Path" value="/path/to/file.jpg" schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Path')).toHaveValue('/path/to/file.jpg');
    });

    it('calls onChange with new value', () => {
      const onChange = vi.fn();
      render(<StringInput label="Path" value="" schema={schema} onChange={onChange} />);

      const input = screen.getByLabelText('Path');
      fireEvent.change(input, { target: { value: 'new/path.jpg' } });

      expect(onChange).toHaveBeenCalledWith('new/path.jpg');
    });

    it('shows placeholder from default', () => {
      render(<StringInput label="Path" value="" schema={schema} onChange={() => {}} />);
      expect(screen.getByPlaceholderText('default.jpg')).toBeInTheDocument();
    });
  });

  describe('EnumDropdown', () => {
    const schema: PropertySchema = {
      type: 'enum',
      enum_values: ['low', 'medium', 'high'],
      description: 'Quality level',
    };

    it('renders all options', () => {
      render(<EnumDropdown label="Quality" value="medium" schema={schema} onChange={() => {}} />);

      expect(screen.getByRole('option', { name: 'low' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'medium' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'high' })).toBeInTheDocument();
    });

    it('selects the current value', () => {
      render(<EnumDropdown label="Quality" value="high" schema={schema} onChange={() => {}} />);
      expect(screen.getByLabelText('Quality')).toHaveValue('high');
    });

    it('calls onChange with selected value', () => {
      const onChange = vi.fn();
      render(<EnumDropdown label="Quality" value="low" schema={schema} onChange={onChange} />);

      fireEvent.change(screen.getByLabelText('Quality'), { target: { value: 'high' } });

      expect(onChange).toHaveBeenCalledWith('high');
    });
  });

  describe('JsonEditor', () => {
    const schema: PropertySchema = {
      type: 'json',
      description: 'Custom configuration',
    };

    it('renders JSON as formatted text', () => {
      const value = { key: 'value', nested: { a: 1 } };
      render(<JsonEditor label="Config" value={value} schema={schema} onChange={() => {}} />);

      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveValue(JSON.stringify(value, null, 2));
    });

    it('calls onChange with parsed JSON', () => {
      const onChange = vi.fn();
      render(<JsonEditor label="Config" value={{}} schema={schema} onChange={onChange} />);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: '{"newKey": "newValue"}' } });

      expect(onChange).toHaveBeenCalledWith({ newKey: 'newValue' });
    });

    it('shows error for invalid JSON', () => {
      render(<JsonEditor label="Config" value={{}} schema={schema} onChange={() => {}} />);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'not valid json' } });

      expect(screen.getByText('Invalid JSON')).toBeInTheDocument();
    });
  });
});
