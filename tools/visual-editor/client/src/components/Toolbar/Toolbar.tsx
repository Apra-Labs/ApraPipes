import { Play, Square, Save, FolderOpen, FileJson, Settings } from 'lucide-react';

/**
 * Main toolbar component with pipeline controls
 */
export function Toolbar() {
  return (
    <header className="h-12 border-b border-border bg-background flex items-center px-4 gap-2">
      {/* Logo/Title */}
      <div className="flex items-center gap-2 mr-4">
        <span className="font-bold text-lg">ApraPipes Studio</span>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-border" />

      {/* File Operations */}
      <div className="flex items-center gap-1 ml-2">
        <ToolbarButton icon={<FolderOpen className="w-4 h-4" />} label="Open" />
        <ToolbarButton icon={<Save className="w-4 h-4" />} label="Save" />
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-border" />

      {/* Pipeline Controls */}
      <div className="flex items-center gap-1 ml-2">
        <ToolbarButton
          icon={<Play className="w-4 h-4" />}
          label="Run"
          variant="success"
        />
        <ToolbarButton
          icon={<Square className="w-4 h-4" />}
          label="Stop"
          variant="danger"
          disabled
        />
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-border" />

      {/* View Controls */}
      <div className="flex items-center gap-1 ml-2">
        <ViewToggleButton active>Visual</ViewToggleButton>
        <ViewToggleButton>JSON</ViewToggleButton>
        <ViewToggleButton>Split</ViewToggleButton>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right side */}
      <div className="flex items-center gap-1">
        <ToolbarButton icon={<FileJson className="w-4 h-4" />} label="Validate" />
        <ToolbarButton icon={<Settings className="w-4 h-4" />} label="Settings" />
      </div>
    </header>
  );
}

interface ToolbarButtonProps {
  icon: React.ReactNode;
  label: string;
  variant?: 'default' | 'success' | 'danger';
  disabled?: boolean;
  onClick?: () => void;
}

function ToolbarButton({
  icon,
  label,
  variant = 'default',
  disabled = false,
  onClick,
}: ToolbarButtonProps) {
  const variantClasses = {
    default: 'hover:bg-muted',
    success: 'hover:bg-green-100 text-green-700',
    danger: 'hover:bg-red-100 text-red-700',
  };

  return (
    <button
      className={`
        flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium
        transition-colors
        ${variantClasses[variant]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
      disabled={disabled}
      onClick={onClick}
      title={label}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

interface ViewToggleButtonProps {
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
}

function ViewToggleButton({ children, active, onClick }: ViewToggleButtonProps) {
  return (
    <button
      className={`
        px-3 py-1 text-sm font-medium rounded transition-colors
        ${active ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}
      `}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
