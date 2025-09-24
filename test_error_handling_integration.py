"""
Integration Tests for Manual Selection Error Handling

Tests the complete error handling system integration with manual selection components.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from manual_selection_errors import (
    ManualSelectionValidator, ErrorRecoveryManager, UserFeedbackManager,
    SelectionValidationError, ShapeDetectionError, EnhancedAnalysisError,
    create_error_context, validate_manual_selection_operation
)
from manual_selection_engine import ManualSelectionEngine
import config


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test integration of error handling with manual selection components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.valid_selection = (50, 50, 100, 100)
        self.invalid_selection = (10, 10, 5, 5)  # Too small
        
        # Create components
        self.validator = ManualSelectionValidator()
        self.error_recovery = ErrorRecoveryManager()
        self.feedback_manager = UserFeedbackManager()
        self.selection_engine = ManualSelectionEngine()
    
    def test_configuration_integration(self):
        """Test that all components use configuration parameters correctly."""
        # Test validator configuration
        self.assertEqual(self.validator.min_selection_size, 
                        getattr(config, 'MIN_SELECTION_SIZE_PX', 20))
        
        # Test error recovery configuration
        self.assertEqual(self.error_recovery.max_retries, 
                        getattr(config, 'MAX_SELECTION_RETRIES', 3))
        
        # Test feedback manager configuration
        self.assertEqual(self.feedback_manager.display_time_ms, 
                        getattr(config, 'ERROR_MESSAGE_DISPLAY_TIME_MS', 3000))
        
        # Test selection engine configuration
        self.assertEqual(self.selection_engine.min_selection_size, 20)
    
    def test_validation_error_flow(self):
        """Test complete validation error handling flow."""
        context = create_error_context(
            "test_validation",
            selection_rect=self.invalid_selection,
            image_shape=self.test_image.shape
        )
        
        # This should raise a validation error
        with self.assertRaises(SelectionValidationError):
            validate_manual_selection_operation(self.test_image, self.invalid_selection)
    
    def test_error_recovery_with_feedback(self):
        """Test error recovery with user feedback integration."""
        error = SelectionValidationError("Test error", self.invalid_selection)
        context = create_error_context(
            "test_recovery",
            selection_rect=self.invalid_selection,
            image_shape=self.test_image.shape
        )
        
        # Handle error with recovery
        recovery_result = self.error_recovery.handle_selection_error(error, context)
        self.assertIsNotNone(recovery_result)
        
        # Add feedback message
        self.feedback_manager.add_error_message(error, context)
        
        # Check that message is available
        message = self.feedback_manager.get_current_message()
        self.assertIsNotNone(message)
        self.assertEqual(self.feedback_manager.get_current_message_type(), "error")
    
    def test_selection_engine_error_integration(self):
        """Test selection engine integration with error handling."""
        # Set up selection engine with error callback
        error_messages = []
        
        def error_callback(error):
            error_messages.append(error)
        
        self.selection_engine.error_callback = error_callback
        
        # Start a selection
        self.selection_engine.start_selection(10, 10)
        self.selection_engine.update_selection(15, 15)  # Very small selection
        
        # Complete selection (should trigger validation error)
        result = self.selection_engine.complete_selection()
        
        # Check that error was handled
        self.assertIsNone(result)  # Should return None due to validation error
        self.assertGreater(len(error_messages), 0)  # Error callback should be called
    
    def test_fallback_strategy_integration(self):
        """Test fallback strategy integration."""
        error = EnhancedAnalysisError("Enhanced analysis failed", fallback_available=True)
        context = create_error_context("test_fallback")
        
        # Handle error with fallback
        recovery_result = self.error_recovery.handle_selection_error(error, context)
        
        self.assertIsNotNone(recovery_result)
        self.assertTrue(recovery_result.get("use_fallback", False))
        self.assertEqual(recovery_result["recovery_method"], "standard_analysis_fallback")
    
    def test_comprehensive_error_scenario(self):
        """Test a comprehensive error scenario with multiple recovery attempts."""
        # Simulate multiple errors in sequence
        errors = [
            SelectionValidationError("Selection too small", self.invalid_selection),
            ShapeDetectionError("No shape found", "circle"),
            EnhancedAnalysisError("Analysis failed", fallback_available=True)
        ]
        
        context = create_error_context(
            "comprehensive_test",
            selection_rect=self.invalid_selection,
            image_shape=self.test_image.shape
        )
        
        recovery_results = []
        for error in errors:
            result = self.error_recovery.handle_selection_error(error, context)
            recovery_results.append(result)
            
            # Add feedback for each error
            self.feedback_manager.add_error_message(error, context)
        
        # Check that all errors were handled
        self.assertEqual(len(recovery_results), 3)
        self.assertTrue(all(r is not None for r in recovery_results))
        
        # Check recovery statistics
        stats = self.error_recovery.get_recovery_stats()
        self.assertEqual(stats["total_errors"], 3)
        self.assertEqual(stats["recovered_errors"], 3)
        self.assertEqual(stats["fallback_uses"], 1)  # Only the enhanced analysis error uses fallback
    
    def test_error_message_prioritization(self):
        """Test error message prioritization in feedback manager."""
        from manual_selection_errors import ErrorSeverity
        
        # Add messages with different priorities
        low_error = SelectionValidationError("Low priority error")
        high_error = ShapeDetectionError("High priority error")
        
        context = create_error_context("priority_test")
        
        # Add low priority first
        self.feedback_manager.add_error_message(low_error, context, ErrorSeverity.LOW)
        
        # Add high priority
        self.feedback_manager.add_error_message(high_error, context, ErrorSeverity.HIGH)
        
        # High priority should be current
        current_message = self.feedback_manager.get_current_message()
        self.assertEqual(current_message, high_error.user_message)
    
    def test_configuration_parameter_validation(self):
        """Test validation of configuration parameters."""
        # Test that validator validates its own configuration
        try:
            self.validator.validate_configuration()
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def test_error_context_creation_utility(self):
        """Test error context creation utility function."""
        context = create_error_context(
            "utility_test",
            selection_rect=self.valid_selection,
            image_shape=self.test_image.shape,
            mode="circle",
            retry_count=1
        )
        
        self.assertEqual(context.operation, "utility_test")
        self.assertEqual(context.selection_rect, self.valid_selection)
        self.assertEqual(context.image_shape, self.test_image.shape)
        self.assertEqual(context.mode, "circle")
        self.assertEqual(context.retry_count, 1)
        self.assertGreater(context.timestamp, 0)


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test error handling for configuration-related issues."""
    
    def test_missing_configuration_parameters(self):
        """Test handling of missing configuration parameters."""
        # Test that components handle missing config gracefully
        with patch.object(config, 'MIN_SELECTION_SIZE_PX', None):
            validator = ManualSelectionValidator()
            # Should use default value
            self.assertEqual(validator.min_selection_size, 20)
    
    def test_invalid_configuration_parameters(self):
        """Test handling of invalid configuration parameters."""
        # Test with invalid minimum selection size
        with patch.object(config, 'MIN_SELECTION_SIZE_PX', -10):
            validator = ManualSelectionValidator()
            
            # Should raise configuration error when validating
            from manual_selection_errors import ConfigurationError
            with self.assertRaises(ConfigurationError):
                validator.validate_configuration()


if __name__ == '__main__':
    unittest.main(verbosity=2)