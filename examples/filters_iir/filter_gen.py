#!/usr/bin/env python
import sys
import json
from filterdesign import design_butterworth, butter_bandpass, butter_highpass, butter_lowpass

def print_usage():
    """Print usage instructions for the CLI tool."""
    print("Butterworth Filter Design CLI Tool")
    print("Usage:")
    print("  lowpass <order> <cutoff> <fs>")
    print("  highpass <order> <cutoff> <fs>")
    print("  bandpass <order> <low_cutoff> <high_cutoff> <fs>")
    print("")
    print("Arguments:")
    print("  order:       Filter order (integer)")
    print("  cutoff:      Cutoff frequency in Hz (float)")
    print("  low_cutoff:  Lower cutoff frequency in Hz (float, bandpass only)")
    print("  high_cutoff: Upper cutoff frequency in Hz (float, bandpass only)")
    print("  fs:          Sampling frequency in Hz (float)")
    print("")
    print("Output: JSON formatted filter coefficients in second-order sections format")
    print("Example: python filter_cli.py lowpass 4 100 1000")

def parse_args():
    """Parse command line arguments."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    filter_type = sys.argv[1].lower()
    
    try:
        if filter_type == "lowpass":
            if len(sys.argv) != 5:
                raise ValueError("Lowpass filter requires exactly 3 parameters")
            order = int(sys.argv[2])
            cutoff = float(sys.argv[3])
            fs = float(sys.argv[4])
            return {"type": filter_type, "order": order, "cutoff": cutoff, "fs": fs}
        
        elif filter_type == "highpass":
            if len(sys.argv) != 5:
                raise ValueError("Highpass filter requires exactly 3 parameters")
            order = int(sys.argv[2])
            cutoff = float(sys.argv[3])
            fs = float(sys.argv[4])
            return {"type": filter_type, "order": order, "cutoff": cutoff, "fs": fs}
        
        elif filter_type == "bandpass":
            if len(sys.argv) != 6:
                raise ValueError("Bandpass filter requires exactly 4 parameters")
            order = int(sys.argv[2])
            low_cutoff = float(sys.argv[3])
            high_cutoff = float(sys.argv[4])
            fs = float(sys.argv[5])
            return {"type": filter_type, "order": order, "cutoff": (low_cutoff, high_cutoff), "fs": fs}
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    except (ValueError, IndexError) as e:
        print(f"Error: {str(e)}")
        print_usage()
        sys.exit(1)

def design_filter(args):
    """Design filter based on arguments."""
    if args["type"] == "lowpass":
        sos = butter_lowpass(args["order"], args["cutoff"], args["fs"])
    elif args["type"] == "highpass":
        sos = butter_highpass(args["order"], args["cutoff"], args["fs"])
    elif args["type"] == "bandpass":
        sos = butter_bandpass(args["order"], args["cutoff"], args["fs"])
    else:
        raise ValueError(f"Unknown filter type: {args['type']}")
    
    return sos

def format_output(sos):
    """Format filter coefficients"""

    formatted_sos = []
    for section in sos:
        formatted_sos += section
    
    return formatted_sos

def main():
    """Main function."""
    args = parse_args()
    try:
        sos = design_filter(args)
        print(format_output(sos))
    except Exception as e:
        print(f"Error designing filter: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
