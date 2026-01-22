import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModulePalette } from '../components/Panels/ModulePalette';
import type { ModuleSchema } from '../types/schema';

const mockModules: Record<string, ModuleSchema> = {
  TestSignalGenerator: {
    category: 'source',
    description: 'Generates test video signals',
    inputs: [],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {},
  },
  ResizeModule: {
    category: 'transform',
    description: 'Resizes images',
    inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
    outputs: [{ name: 'output', frame_types: ['RAW_IMAGE'] }],
    properties: {},
  },
  FileWriterModule: {
    category: 'sink',
    description: 'Writes frames to files',
    inputs: [{ name: 'input', frame_types: ['RAW_IMAGE'] }],
    outputs: [],
    properties: {},
  },
};

describe('ModulePalette', () => {
  it('renders category groups', () => {
    render(<ModulePalette modules={mockModules} />);

    expect(screen.getByText('source')).toBeInTheDocument();
    expect(screen.getByText('transform')).toBeInTheDocument();
    expect(screen.getByText('sink')).toBeInTheDocument();
  });

  it('displays module names', () => {
    render(<ModulePalette modules={mockModules} />);

    expect(screen.getByText('TestSignalGenerator')).toBeInTheDocument();
    expect(screen.getByText('ResizeModule')).toBeInTheDocument();
    expect(screen.getByText('FileWriterModule')).toBeInTheDocument();
  });

  it('displays module descriptions', () => {
    render(<ModulePalette modules={mockModules} />);

    expect(screen.getByText('Generates test video signals')).toBeInTheDocument();
    expect(screen.getByText('Resizes images')).toBeInTheDocument();
  });

  it('filters modules by search query', () => {
    render(<ModulePalette modules={mockModules} />);

    const searchInput = screen.getByPlaceholderText('Search modules...');
    fireEvent.change(searchInput, { target: { value: 'signal' } });

    expect(screen.getByText('TestSignalGenerator')).toBeInTheDocument();
    expect(screen.queryByText('ResizeModule')).not.toBeInTheDocument();
    expect(screen.queryByText('FileWriterModule')).not.toBeInTheDocument();
  });

  it('shows no modules message when search has no results', () => {
    render(<ModulePalette modules={mockModules} />);

    const searchInput = screen.getByPlaceholderText('Search modules...');
    fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

    expect(screen.getByText('No modules found')).toBeInTheDocument();
  });

  it('collapses and expands category groups', () => {
    render(<ModulePalette modules={mockModules} />);

    // Source category should be expanded by default
    expect(screen.getByText('TestSignalGenerator')).toBeInTheDocument();

    // Click to collapse
    const sourceHeader = screen.getByText('source');
    fireEvent.click(sourceHeader);

    // Module should no longer be visible
    expect(screen.queryByText('TestSignalGenerator')).not.toBeInTheDocument();

    // Click to expand again
    fireEvent.click(sourceHeader);

    // Module should be visible again
    expect(screen.getByText('TestSignalGenerator')).toBeInTheDocument();
  });

  it('module cards are draggable', () => {
    render(<ModulePalette modules={mockModules} />);

    const moduleCard = screen.getByText('TestSignalGenerator').closest('[draggable]');
    expect(moduleCard).toHaveAttribute('draggable', 'true');
  });
});
