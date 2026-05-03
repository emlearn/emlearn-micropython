#!/usr/bin/env python3
"""
Helper script to run tests on QEMU MicroPython using mpremote.
Starts QEMU with a PTY, mounts local directory, runs test script, and cleans up.

Usage:
    python3 run_qemu_tests.py [options]

Options:
    --test TEST_SCRIPT     Test script to run (default: tests/test_all.py)
    --mount DIR            Local directory to mount on device
    --board BOARD          QEMU board to use (default: MPS2_AN500)
    --arch ARCH            MicroPython architecture (default: armv7emdp)
    --abi-version VERSION  MicroPython ABI version (default: 6.3)
    --modules DIR          MicroPython modules directory
    --timeout SECONDS      Timeout for tests (default: 120)
    --debug               Enable debug output

Examples:
    python3 run_qemu_tests.py --board MPS2_AN500 --mount .
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def debug_print(msg):
    """Print debug message and flush."""
    print(f"DEBUG: {msg}")
    sys.stdout.flush()


def run_tests(pty_path, test_script, mount_path, modules_path, remote_modules):
    """Run the test script on QEMU using mpremote mount.
    
    Reads output line by line until TEST END marker is found.
    """
    test_script_path = Path(test_script)
    
    debug_print(f"test_script: {test_script}")
    debug_print(f"mount_path: {mount_path}")
    
    debug_print(f"Mounting {mount_path} on device...")
    debug_print(f"Running {test_script}...")
    print(f"Mounting {mount_path} on device...")
    print(f"Running {test_script_path}...")
    sys.stdout.flush()
    
    cmd = [
        'mpremote', 'connect', pty_path,
        'mount', str(mount_path),
        'run', str(test_script_path),
    ]
    
    debug_print(f"Running command: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    output_lines = []
    test_complete = False
    return_code = 1
    
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            debug_print(f"Process ended, poll={proc.poll()}")
            break
        
        if line:
            output_lines.append(line)
            print(line, end='')
            sys.stdout.flush()
            
            if '=== TEST END ===' in line:
                test_complete = True
                stderr = proc.stderr.read()
                if stderr:
                    print("STDERR:", stderr, file=sys.stderr)
                proc.wait()
                return 0
        
        if proc.poll() is not None:
            debug_print(f"Process poll check: {proc.poll()}")
            break
    
    stderr = proc.stderr.read()
    if stderr:
        print("STDERR:", stderr, file=sys.stderr)
        sys.stderr.flush()
    
    if not test_complete:
        print("\nError: Tests did not complete (no TEST END marker)", file=sys.stderr)
        print(f"Output so far: {len(output_lines)} lines", file=sys.stderr)
        for line in output_lines[-20:]:
            print(f"  {line.rstrip()}", file=sys.stderr)
    
    return return_code


def main():
    parser = argparse.ArgumentParser(
        description='Run tests on QEMU MicroPython via mpremote',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test', '-t', default='tests/test_all.py',
                        help='Test script to run (default: tests/test_all.py)')
    parser.add_argument('--mount', '-m',
                        help='Local directory to mount (default: cwd)')
    parser.add_argument('--board', '-b', default='MPS2_AN500',
                        help='QEMU board name (default: MPS2_AN500)')
    parser.add_argument('--arch', '-a', default='armv7emdp',
                        help='MicroPython architecture (default: armv7emdp)')
    parser.add_argument('--abi-version', '-v', default='6.3',
                        help='MicroPython ABI version (default: 6.3)')
    parser.add_argument('--modules',
                        help='MicroPython modules directory (default: auto-detect from dist/)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds (default: 120)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    script_dir = Path(__file__).parent.parent
    mount_path = Path(args.mount) if args.mount else script_dir
    
    debug_print(f"script_dir: {script_dir}")
    debug_print(f"mount_path: {mount_path}")
    
    # Find MicroPython modules directory
    module_dir = f'{args.arch}_{args.abi_version}'
    
    if args.modules:
        modules_path = str(Path(args.modules).absolute())
    else:
        candidates = list((script_dir / 'dist').glob(module_dir))
        if candidates:
            modules_path = str(candidates[0])
            print(f"Auto-detected modules: {modules_path}")
        else:
            modules_path = ''
            print(f"Warning: Could not find {module_dir} in dist/", file=sys.stderr)
    
    debug_print(f"module_dir: {module_dir}")
    debug_print(f"modules_path: {modules_path}")
    
    remote_modules = ''
    if modules_path:
        try:
            modules_rel = Path(modules_path).relative_to(mount_path)
        except ValueError:
            modules_rel = Path(modules_path).name
        remote_modules = f'/remote/{modules_rel}'
    
    debug_print(f"remote_modules: {remote_modules}")
    
    # Find QEMU port directory and firmware
    qemu_port_dir = script_dir / 'dependencies' / 'micropython' / 'ports' / 'qemu'
    if not qemu_port_dir.exists():
        print(f"Error: QEMU port directory not found: {qemu_port_dir}", file=sys.stderr)
        return 1
    
    firmware_path = qemu_port_dir / f'build-{args.board}' / 'firmware.elf'
    if not firmware_path.exists():
        print(f"Error: Firmware not found: {firmware_path}", file=sys.stderr)
        print(f"Run 'make qemu_build QEMU_BOARD={args.board}' first", file=sys.stderr)
        return 1
    
    qemu_machine = {
        'MPS2_AN500': 'mps2-an500',
        'MPS2_AN385': 'mps2-an385',
    }.get(args.board, 'mps2-an500')
    
    qemu_cmd = [
        'qemu-system-arm',
        '-machine', qemu_machine,
        '-nographic',
        '-monitor', 'null',
        '-semihosting',
        '-serial', 'pty',
        '-kernel', str(firmware_path)
    ]
    
    print(f"Starting QEMU with firmware: {firmware_path}")
    print(f"Board: {args.board}, Arch: {args.arch}, ABI: {args.abi_version}")
    sys.stdout.flush()
    
    debug_print(f"QEMU command: {' '.join(qemu_cmd)}")
    
    qemu_proc = subprocess.Popen(
        qemu_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(qemu_port_dir)
    )
    
    output_lines = []
    pty_path = None
    start_time = time.time()
    
    print("Waiting for QEMU to start...")
    sys.stdout.flush()
    
    while time.time() - start_time < 30:
        line = qemu_proc.stdout.readline()
        if line:
            output_lines.append(line)
            print(line, end='')
            sys.stdout.flush()
            match = re.search(r'char device redirected to (/dev/pts/\d+)', line)
            if match:
                pty_path = match.group(1)
                break
        else:
            time.sleep(0.1)
    
    if not pty_path:
        print("Error: Could not find PTY device from QEMU output", file=sys.stderr)
        print("Output:", ''.join(output_lines), file=sys.stderr)
        qemu_proc.terminate()
        return 1
    
    print(f"PTY device: {pty_path}")
    sys.stdout.flush()
    
    print("Waiting for MicroPython to boot...")
    sys.stdout.flush()
    time.sleep(2)
    
    result = run_tests(pty_path, args.test, mount_path, modules_path, remote_modules)
    
    print("Stopping QEMU...")
    sys.stdout.flush()
    qemu_proc.terminate()
    try:
        qemu_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Warning: QEMU did not terminate gracefully, killing...", file=sys.stderr)
        qemu_proc.kill()
        qemu_proc.wait()
    
    return result


if __name__ == '__main__':
    sys.exit(main())
