#!/usr/bin/env python3
"""
Open Memory Suite - Session File Explorer

Interactive script to explore and copy project code organized by development phases.
Each phase represents cohesive components or related files that can be reviewed together.

Usage:
    python show_session_files.py
    
Features:
- Phase-based file organization for logical code review
- Copy-to-clipboard functionality for easy sharing
- File content preview with syntax highlighting
- Project statistics and overview
- Cross-platform clipboard support
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class SessionFileExplorer:
    """Interactive file explorer organized by development phases."""
    
    def __init__(self, project_root: Path):
        """Initialize with project root directory."""
        self.project_root = Path(project_root)
        self.phases = self._define_phases()
        
    def _define_phases(self) -> Dict[str, Dict[str, List[str]]]:
        """Define file organization by development phases."""
        return {
            "1_foundation": {
                "description": "Core architecture and interfaces",
                "files": [
                    "open_memory_suite/adapters/base.py",
                    "open_memory_suite/dispatcher/core.py",
                    "open_memory_suite/benchmark/cost_model.py",
                    "open_memory_suite/benchmark/trace.py",
                    "pyproject.toml"
                ]
            },
            
            "2_adapters": {
                "description": "Memory storage adapters implementation",
                "files": [
                    "open_memory_suite/adapters/memory_store.py",
                    "open_memory_suite/adapters/file_store.py", 
                    "open_memory_suite/adapters/faiss_store.py",
                    "open_memory_suite/adapters/__init__.py"
                ]
            },
            
            "3_dispatcher": {
                "description": "Intelligent routing system and policies",
                "files": [
                    "open_memory_suite/dispatcher/frugal_dispatcher.py",
                    "open_memory_suite/dispatcher/heuristic_policy.py",
                    "open_memory_suite/dispatcher/__init__.py"
                ]
            },
            
            "4_evaluation": {
                "description": "Benchmarking and evaluation framework", 
                "files": [
                    "open_memory_suite/benchmark/harness.py",
                    "open_memory_suite/benchmark/evaluation_wrappers.py",
                    "open_memory_suite/benchmark/cost_model.yaml",
                    "open_memory_suite/benchmark/__init__.py"
                ]
            },
            
            "5_demos": {
                "description": "Demonstration scripts and validation",
                "files": [
                    "demo_frugal_dispatcher.py",
                    "validate_m2_milestone.py"
                ]
            },
            
            "6_tests": {
                "description": "Test suite and quality assurance",
                "files": [
                    "tests/",  # Special handling for directory
                ]
            },
            
            "7_docs": {
                "description": "Documentation and planning",
                "files": [
                    "implementationPlan.md",
                    "M2_MILESTONE_ACHIEVED.md", 
                    "IMPLEMENTATION_PLAN_NEXT.md",
                    "docs/README.md",
                    "docs/future_implementation_roadmap.md",
                    "docs/status_update.md"
                ]
            },
            
            "8_config": {
                "description": "Configuration and deployment",
                "files": [
                    "pyproject.toml",
                    "poetry.lock",
                    ".gitignore",
                    ".gitmodules"
                ]
            }
        }
    
    def copy_to_clipboard(self, content: str) -> bool:
        """Copy content to system clipboard."""
        try:
            # Try multiple clipboard methods for cross-platform support
            if sys.platform == "darwin":  # macOS
                subprocess.run(["pbcopy"], input=content, text=True, check=True)
            elif sys.platform == "win32":  # Windows
                subprocess.run(["clip"], input=content, text=True, check=True)
            else:  # Linux
                try:
                    subprocess.run(["xclip", "-selection", "clipboard"], 
                                 input=content, text=True, check=True)
                except FileNotFoundError:
                    try:
                        subprocess.run(["xsel", "--clipboard", "--input"], 
                                     input=content, text=True, check=True)
                    except FileNotFoundError:
                        print("❌ Clipboard not available. Install xclip or xsel.")
                        return False
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to copy to clipboard")
            return False
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file relative to project root."""
        full_path = self.project_root / file_path
        
        if file_path.endswith("/"):  # Directory handling
            return self._get_directory_listing(full_path)
        
        if not full_path.exists():
            return f"❌ File not found: {file_path}"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add file header for context
            header = f"# File: {file_path}\n# Size: {len(content)} characters\n# Modified: {os.path.getmtime(full_path)}\n\n"
            return header + content
        except Exception as e:
            return f"❌ Error reading {file_path}: {e}"
    
    def _get_directory_listing(self, dir_path: Path) -> str:
        """Get a listing of files in a directory."""
        if not dir_path.exists():
            return f"❌ Directory not found: {dir_path}"
        
        files = []
        for item in sorted(dir_path.rglob("*")):
            if item.is_file() and not item.name.startswith('.'):
                rel_path = item.relative_to(dir_path)
                size = item.stat().st_size
                files.append(f"{rel_path} ({size} bytes)")
        
        header = f"# Directory: {dir_path.relative_to(self.project_root)}\n"
        header += f"# Files: {len(files)}\n\n"
        
        return header + "\n".join(files)
    
    def show_phase_summary(self) -> None:
        """Show overview of all phases."""
        print("\n🎯 Open Memory Suite - Development Phases")
        print("=" * 60)
        
        total_files = 0
        for phase_id, phase_info in self.phases.items():
            file_count = len(phase_info["files"])
            total_files += file_count
            
            print(f"\n📁 {phase_id.upper()}")
            print(f"   {phase_info['description']}")
            print(f"   Files: {file_count}")
            
            # Show first few files as preview
            preview_files = phase_info["files"][:3]
            for file_path in preview_files:
                exists = "✅" if (self.project_root / file_path).exists() else "❌"
                print(f"   {exists} {file_path}")
            
            if len(phase_info["files"]) > 3:
                print(f"   ... and {len(phase_info['files']) - 3} more files")
        
        print(f"\n📊 Total Files: {total_files}")
        print(f"📂 Project Root: {self.project_root}")
    
    def show_phase_files(self, phase_id: str, copy_to_clipboard: bool = False) -> None:
        """Show files in a specific phase."""
        if phase_id not in self.phases:
            print(f"❌ Unknown phase: {phase_id}")
            self.show_available_phases()
            return
        
        phase_info = self.phases[phase_id]
        print(f"\n📁 {phase_id.upper()}: {phase_info['description']}")
        print("=" * 60)
        
        all_content = []
        
        for i, file_path in enumerate(phase_info["files"], 1):
            exists = "✅" if (self.project_root / file_path).exists() else "❌"
            print(f"\n{i}. {exists} {file_path}")
            
            if exists == "✅":
                content = self.get_file_content(file_path)
                if content:
                    all_content.append(f"\n{'='*80}\n# PHASE {phase_id.upper()}: {file_path}\n{'='*80}\n")
                    all_content.append(content)
                    
                    # Show preview (first 5 lines)
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:5])
                    if len(lines) > 5:
                        preview += f"\n... ({len(lines) - 5} more lines)"
                    print(f"   Preview:\n{preview}")
        
        if copy_to_clipboard and all_content:
            combined_content = '\n'.join(all_content)
            if self.copy_to_clipboard(combined_content):
                print(f"\n✅ Copied {len(all_content)} files to clipboard ({len(combined_content)} characters)")
            else:
                print(f"\n❌ Failed to copy to clipboard")
    
    def show_file_content(self, file_path: str, copy_to_clipboard: bool = False) -> None:
        """Show content of a specific file."""
        content = self.get_file_content(file_path)
        if not content:
            print(f"❌ Could not read file: {file_path}")
            return
        
        print(f"\n📄 {file_path}")
        print("=" * 60)
        print(content)
        
        if copy_to_clipboard:
            if self.copy_to_clipboard(content):
                print(f"\n✅ Copied to clipboard ({len(content)} characters)")
            else:
                print(f"\n❌ Failed to copy to clipboard")
    
    def show_available_phases(self) -> None:
        """Show list of available phases."""
        print("\n🎯 Available Phases:")
        for phase_id, phase_info in self.phases.items():
            print(f"   {phase_id}: {phase_info['description']}")
    
    def interactive_mode(self) -> None:
        """Run interactive mode for exploring files."""
        print("\n🚀 Open Memory Suite - Interactive File Explorer")
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n💻 > ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif command in ['help', 'h']:
                    self._show_help()
                elif command in ['summary', 'overview', 's']:
                    self.show_phase_summary()
                elif command in ['phases', 'p']:
                    self.show_available_phases()
                elif command.startswith('phase '):
                    parts = command.split()
                    if len(parts) >= 2:
                        phase_id = parts[1]
                        copy_flag = len(parts) > 2 and parts[2] in ['copy', 'c']
                        self.show_phase_files(phase_id, copy_flag)
                    else:
                        print("❌ Usage: phase <phase_id> [copy]")
                elif command.startswith('file '):
                    parts = command.split(maxsplit=1)
                    if len(parts) >= 2:
                        file_path = parts[1]
                        copy_flag = file_path.endswith(' copy') or file_path.endswith(' c')
                        if copy_flag:
                            file_path = file_path.rsplit(' ', 1)[0]
                        self.show_file_content(file_path, copy_flag)
                    else:
                        print("❌ Usage: file <file_path> [copy]")
                else:
                    print(f"❌ Unknown command: {command}")
                    self._show_help()
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n📖 Commands:")
        print("   summary | s          - Show overview of all phases")
        print("   phases | p           - List available phases")
        print("   phase <id> [copy]    - Show files in phase (optionally copy to clipboard)")
        print("   file <path> [copy]   - Show specific file content (optionally copy)")
        print("   help | h             - Show this help")
        print("   quit | q             - Exit")
        print("\n📝 Examples:")
        print("   phase 1_foundation")
        print("   phase 3_dispatcher copy")
        print("   file open_memory_suite/dispatcher/core.py")
        print("   file demo_frugal_dispatcher.py copy")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Open Memory Suite File Explorer")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--phase", help="Show specific phase")
    parser.add_argument("--file", help="Show specific file")
    parser.add_argument("--copy", action="store_true", help="Copy to clipboard")
    parser.add_argument("--summary", action="store_true", help="Show project summary")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    project_root = Path(args.root).resolve()
    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)
    
    explorer = SessionFileExplorer(project_root)
    
    if args.summary:
        explorer.show_phase_summary()
    elif args.phase:
        explorer.show_phase_files(args.phase, args.copy)
    elif args.file:
        explorer.show_file_content(args.file, args.copy)
    elif args.interactive:
        explorer.interactive_mode()
    else:
        # Default to interactive mode
        explorer.show_phase_summary()
        print("\nRun with --interactive for interactive mode, or --help for options")


if __name__ == "__main__":
    main()