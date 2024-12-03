import argparse
import subprocess
import os
import sys

def run_command(command, cwd=None):
    """Run a shell command and print its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            text=True,
            check=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run benchmark with build command.")
    parser.add_argument("benchmark", type=str, help="The benchmark name (folder).")
    parser.add_argument("model", type=str, help="The model name (folder).")
    parser.add_argument(
        "--args", type=str, help="Additional program arguments.", default=""
    )
    parser.add_argument(
        "--build_cmd", type=str, help="Custom build command.", default="make"
    )
    parser.add_argument(
        "--run_cmd", type=str, help="Custom run command.", default="./upmem_test"
    )
    parser.add_argument(
        "--time_output", type=str, help="Timing output file", default="output.run"
    )
    parser.add_argument(
        "--compile_defines", type=str, help="Compiler Defines.", default=""
    )

    args = parser.parse_args()

    # Construct the paths to the benchmark and subcategory folders
    benchmark_path = os.path.join(args.benchmark, args.model)

    if not os.path.exists(benchmark_path):
        print(f"Error: The path '{benchmark_path}' does not exist.")
        sys.exit(1)
    
    # Step 1: Run the build command
    print(f"Building in: {benchmark_path}")
    build_cmd = f"{args.compile_defines} {args.build_cmd}"
    run_command(build_cmd, cwd=benchmark_path)

    # Step 2: Run the benchmark with additional arguments
    print(f"Running benchmark in: {benchmark_path}")
    benchmark_command = f"/usr/bin/time -o {args.time_output} {args.run_cmd} {args.args}"
    print(f"Benchmark command: {benchmark_command}")
    run_command(benchmark_command, cwd=benchmark_path)

if __name__ == "__main__":
    main()
