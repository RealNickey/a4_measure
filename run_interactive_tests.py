#!/usr/bin/env python3
"""
Comprehensive test runner for interactive functionality.

This script runs all test suites for the interactive inspect mode functionality
and provides a detailed summary of results, including performance metrics.
"""

import unittest
import sys
import time
import traceback
from io import StringIO


def run_test_suite(test_module_name, description):
    """
    Run a specific test suite and return results.
    
    Args:
        test_module_name: Name of the test module to import and run
        description: Human-readable description of the test suite
        
    Returns:
        Tuple of (success, results_summary, execution_time)
    """
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import the test module
        test_module = __import__(test_module_name)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests with custom result handler
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            buffer=True
        )
        
        result = runner.run(suite)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results to console
        output = stream.getvalue()
        print(output)
        
        # Create summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        success_count = total_tests - failures - errors - skipped
        
        summary = {
            'total': total_tests,
            'success': success_count,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'execution_time': execution_time,
            'success_rate': (success_count / total_tests * 100) if total_tests > 0 else 0
        }
        
        success = failures == 0 and errors == 0
        
        return success, summary, output
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        error_msg = f"Failed to run {description}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        summary = {
            'total': 0,
            'success': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'execution_time': execution_time,
            'success_rate': 0,
            'error_message': error_msg
        }
        
        return False, summary, error_msg


def print_summary(all_results):
    """Print comprehensive summary of all test results."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = 0
    total_success = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0
    
    print(f"{'Test Suite':<35} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Time':<10} {'Rate':<8}")
    print(f"{'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")
    
    for description, (success, summary, _) in all_results.items():
        status_icon = "✓" if success else "✗"
        
        print(f"{description:<35} "
              f"{summary['total']:<8} "
              f"{summary['success']:<8} "
              f"{summary['failures']:<8} "
              f"{summary['errors']:<8} "
              f"{summary['execution_time']:<9.2f}s "
              f"{summary['success_rate']:<7.1f}%")
        
        total_tests += summary['total']
        total_success += summary['success']
        total_failures += summary['failures']
        total_errors += summary['errors']
        total_skipped += summary['skipped']
        total_time += summary['execution_time']
    
    print(f"{'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")
    overall_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    print(f"{'TOTAL':<35} "
          f"{total_tests:<8} "
          f"{total_success:<8} "
          f"{total_failures:<8} "
          f"{total_errors:<8} "
          f"{total_time:<9.2f}s "
          f"{overall_rate:<7.1f}%")
    
    print(f"\nOverall Results:")
    print(f"  • Total test cases: {total_tests}")
    print(f"  • Successful: {total_success} ({overall_rate:.1f}%)")
    print(f"  • Failed: {total_failures}")
    print(f"  • Errors: {total_errors}")
    print(f"  • Skipped: {total_skipped}")
    print(f"  • Total execution time: {total_time:.2f} seconds")
    
    # Performance summary
    print(f"\nPerformance Highlights:")
    for description, (success, summary, output) in all_results.items():
        if "Performance" in description:
            print(f"  • {description}: {summary['execution_time']:.2f}s")
            # Extract performance metrics from output if available
            if "Average" in output:
                lines = output.split('\n')
                for line in lines:
                    if "Average" in line and ("ms" in line or "MB" in line):
                        print(f"    - {line.strip()}")
    
    # Failure summary
    if total_failures > 0 or total_errors > 0:
        print(f"\nFailure Summary:")
        for description, (success, summary, output) in all_results.items():
            if not success:
                print(f"  • {description}: {summary['failures']} failures, {summary['errors']} errors")
                if 'error_message' in summary:
                    print(f"    Error: {summary['error_message'][:200]}...")
    
    return total_failures == 0 and total_errors == 0


def main():
    """Main test runner function."""
    print("Interactive Functionality Test Suite")
    print("====================================")
    print("This comprehensive test suite validates all aspects of the interactive inspect mode.")
    print("Testing: hit testing accuracy, coordinate transformation, selection priority,")
    print("         hover-click workflow, and real-time performance responsiveness.")
    
    # Define test suites to run
    test_suites = [
        ("test_hit_testing", "Hit Testing Engine"),
        ("test_hit_testing_integration", "Hit Testing Integration"),
        ("test_interaction_state", "Interaction State Management"),
        ("test_coordinate_transformation", "Coordinate Transformation"),
        ("test_selection_priority", "Selection Priority Logic"),
        ("test_hover_click_workflow", "Hover-Click Workflow"),
        ("test_performance_responsiveness", "Performance & Responsiveness"),
        ("test_edge_cases_integration", "Edge Cases & Integration")
    ]
    
    all_results = {}
    overall_start_time = time.time()
    
    # Run each test suite
    for module_name, description in test_suites:
        success, summary, output = run_test_suite(module_name, description)
        all_results[description] = (success, summary, output)
    
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    
    # Print comprehensive summary
    overall_success = print_summary(all_results)
    
    print(f"\nTotal test execution time: {total_execution_time:.2f} seconds")
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print("The interactive functionality is ready for production use.")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED ❌")
        print("Please review the failures above and fix the issues.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)