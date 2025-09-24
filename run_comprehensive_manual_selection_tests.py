"""
Comprehensive test runner for manual shape selection validation.

This script runs all test suites for manual selection validation including:
- Accuracy validation across various scenarios
- Performance testing
- Measurement accuracy comparison
- Edge case handling

Requirements tested: 1.4, 1.5, 4.4, 4.5
"""

import sys
import os
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any

def run_test_suite(test_file: str, description: str) -> Dict[str, Any]:
    """Run a test suite and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"FILE: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results from output
        output_lines = result.stdout.split('\n')
        error_lines = result.stderr.split('\n')
        
        # Look for test summary information
        tests_run = 0
        failures = 0
        errors = 0
        
        for line in output_lines:
            if "Tests run:" in line:
                try:
                    tests_run = int(line.split("Tests run:")[1].split()[0])
                except:
                    pass
            elif "Failures:" in line:
                try:
                    failures = int(line.split("Failures:")[1].split()[0])
                except:
                    pass
            elif "Errors:" in line:
                try:
                    errors = int(line.split("Errors:")[1].split()[0])
                except:
                    pass
        
        return {
            "test_file": test_file,
            "description": description,
            "success": result.returncode == 0,
            "duration": duration,
            "tests_run": tests_run,
            "failures": failures,
            "errors": errors,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "test_file": test_file,
            "description": description,
            "success": False,
            "duration": 300,
            "tests_run": 0,
            "failures": 0,
            "errors": 1,
            "stdout": "",
            "stderr": "Test timed out after 5 minutes",
            "return_code": -1
        }
    except Exception as e:
        return {
            "test_file": test_file,
            "description": description,
            "success": False,
            "duration": 0,
            "tests_run": 0,
            "failures": 0,
            "errors": 1,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive test report."""
    report_lines = []
    
    # Header
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE MANUAL SELECTION TEST REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    total_suites = len(results)
    successful_suites = sum(1 for r in results if r["success"])
    total_tests = sum(r["tests_run"] for r in results)
    total_failures = sum(r["failures"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_duration = sum(r["duration"] for r in results)
    
    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Test Suites Run: {total_suites}")
    report_lines.append(f"Successful Suites: {successful_suites}")
    report_lines.append(f"Failed Suites: {total_suites - successful_suites}")
    report_lines.append(f"Total Tests: {total_tests}")
    report_lines.append(f"Total Failures: {total_failures}")
    report_lines.append(f"Total Errors: {total_errors}")
    report_lines.append(f"Total Duration: {total_duration:.1f} seconds")
    report_lines.append(f"Overall Success Rate: {(total_tests - total_failures - total_errors) / max(total_tests, 1) * 100:.1f}%")
    report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 40)
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        report_lines.append(f"{status} {result['description']}")
        report_lines.append(f"  File: {result['test_file']}")
        report_lines.append(f"  Duration: {result['duration']:.1f}s")
        report_lines.append(f"  Tests: {result['tests_run']}, Failures: {result['failures']}, Errors: {result['errors']}")
        
        if not result["success"]:
            report_lines.append(f"  Return Code: {result['return_code']}")
            if result["stderr"]:
                report_lines.append(f"  Error: {result['stderr'][:200]}...")
        
        report_lines.append("")
    
    # Performance analysis
    if any(r["success"] for r in results):
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        
        performance_results = [r for r in results if "performance" in r["test_file"].lower() and r["success"]]
        if performance_results:
            report_lines.append("Performance test suites completed successfully:")
            for result in performance_results:
                report_lines.append(f"  - {result['description']}: {result['duration']:.1f}s")
        
        accuracy_results = [r for r in results if "accuracy" in r["test_file"].lower() and r["success"]]
        if accuracy_results:
            report_lines.append("Accuracy test suites completed successfully:")
            for result in accuracy_results:
                report_lines.append(f"  - {result['description']}: {result['duration']:.1f}s")
        
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if total_failures == 0 and total_errors == 0:
        report_lines.append("✅ All tests passed! Manual selection system is ready for production.")
    else:
        report_lines.append("⚠️  Issues found that need attention:")
        
        if total_failures > 0:
            report_lines.append(f"  - {total_failures} test failures indicate functionality issues")
        
        if total_errors > 0:
            report_lines.append(f"  - {total_errors} test errors indicate system/setup issues")
        
        failed_suites = [r for r in results if not r["success"]]
        if failed_suites:
            report_lines.append("  - Failed test suites:")
            for result in failed_suites:
                report_lines.append(f"    * {result['description']}")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    return "\n".join(report_lines)


def main():
    """Main function to run comprehensive test suite."""
    print("COMPREHENSIVE MANUAL SELECTION TEST SUITE")
    print("="*60)
    print("This will run all test suites for manual selection validation")
    print("Estimated time: 5-10 minutes")
    print("")
    
    # Define test suites to run
    test_suites = [
        {
            "file": "test_manual_selection_comprehensive.py",
            "description": "Comprehensive Manual Selection Tests"
        },
        {
            "file": "test_manual_selection_performance.py", 
            "description": "Performance and Responsiveness Tests"
        },
        {
            "file": "test_measurement_accuracy_comparison.py",
            "description": "Measurement Accuracy Validation"
        },
        {
            "file": "test_manual_selection_integration.py",
            "description": "Integration Tests (if available)"
        },
        {
            "file": "test_enhanced_contour_analyzer.py",
            "description": "Enhanced Contour Analyzer Tests"
        },
        {
            "file": "test_shape_snapping_engine.py",
            "description": "Shape Snapping Engine Tests"
        }
    ]
    
    # Check which test files exist
    available_tests = []
    for test_suite in test_suites:
        if os.path.exists(test_suite["file"]):
            available_tests.append(test_suite)
        else:
            print(f"⚠️  Test file not found: {test_suite['file']}")
    
    if not available_tests:
        print("❌ No test files found! Please ensure test files are in the current directory.")
        return 1
    
    print(f"\nFound {len(available_tests)} test suites to run:")
    for test_suite in available_tests:
        print(f"  - {test_suite['description']}")
    
    # Ask for confirmation
    response = input(f"\nProceed with running {len(available_tests)} test suites? (y/n): ").lower()
    if response != 'y':
        print("Test run cancelled.")
        return 0
    
    # Run all test suites
    results = []
    start_time = time.time()
    
    for i, test_suite in enumerate(available_tests, 1):
        print(f"\n[{i}/{len(available_tests)}] Starting: {test_suite['description']}")
        result = run_test_suite(test_suite["file"], test_suite["description"])
        results.append(result)
        
        # Print immediate result
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"[{i}/{len(available_tests)}] {status} {test_suite['description']} ({result['duration']:.1f}s)")
        
        if not result["success"] and result["stderr"]:
            print(f"    Error: {result['stderr'][:100]}...")
    
    total_time = time.time() - start_time
    
    # Generate and display report
    report = generate_report(results)
    print("\n" + report)
    
    # Save report to file
    report_filename = f"manual_selection_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"\n📄 Full report saved to: {report_filename}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    # Return appropriate exit code
    all_passed = all(r["success"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)