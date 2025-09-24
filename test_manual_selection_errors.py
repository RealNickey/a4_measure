"""
Unit Tests for Manual Selection Error Handling

Tests error handling, validation, recovery mechanisms, and user feedback
for manual shape selection operations.

Requirements addressed: 4.3, 2.3
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from manual_selection_errors import (
    ManualSelectionError, SelectionValidationError, ShapeDetectionError,
    EnhancedAnalysisError, ConfigurationError, TimeoutError,
    ErrorSeverity, ErrorContext, ManualSelectionValidator,
    ErrorRecoveryManager, UserFeedbackManager,
    create_error_context, validate_manual_selection_operation,
    handle_operation_with_recovery
)
import config


class TestManualSelectionErrors(unittest.TestCase):
    """Test custom exception classes for manual selection."""
    
    def test_manual_selection_error_creation(self):
        """Test basic ManualSelectionError creation."""
        error = ManualSelectionError("Test error", "TEST_CODE", True, "User message")
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.error_code, "TEST_CODE")
        self.assertTrue(error.recoverable)
        self.assertEqual(error.user_message, "User message")
        self.assertIsInstance(error.timestamp, float)
    
    def test_selection_validation_error(self):
        """Test SelectionValidationError with selection rectangle."""
        selection_rect = (10, 10, 50, 50)
        error = SelectionValidationError("Invalid selection", selection_rect)
        
        self.assertEqual(error.error_code, "SELECTION_INVALID")
        self.assertTrue(error.recoverable)
        self.assertEqual(error.selection_rect, selection_rect)
        self.assertIn("invalid", error.user_message.lower())
    
    def test_shape_detection_error(self):
        """Test ShapeDetectionError with shape type."""
        error = ShapeDetectionError("No circle found", "circle")
        
        self.assertEqual(error.error_code, "SHAPE_NOT_FOUND")
        self.assertTrue(error.recoverable)
        self.assertEqual(error.shape_type, "circle")
        self.assertIn("circle", error.user_message.lower())
    
    def test_enhanced_analysis_error(self):
        """Test EnhancedAnalysisError with fallback availability."""
        error = EnhancedAnalysisError("Analysis failed", fallback_available=True)
        
        self.assertEqual(error.error_code, "ENHANCED_ANALYSIS_FAILED")
        self.assertTrue(error.recoverable)
        self.assertTrue(error.fallback_available)
    
    def test_configuration_error(self):
        """Test ConfigurationError with parameter name."""
        error = ConfigurationError("Invalid config", "TEST_PARAM")
        
        self.assertEqual(error.error_code, "CONFIG_INVALID")
        self.assertFalse(error.recoverable)
        self.assertEqual(error.parameter_name, "TEST_PARAM")
    
    def test_timeout_error(self):
        """Test TimeoutError with timeout value."""
        error = TimeoutError("Operation timed out", 5000)
        
        self.assertEqual(error.error_code, "OPERATION_TIMEOUT")
        self.assertTrue(error.recoverable)
        self.assertEqual(error.timeout_ms, 5000)
        self.assertIn("5000", error.user_message)


class TestErrorContext(unittest.TestCase):
    """Test ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation with default values."""
        context = ErrorContext("test_operation")
        
        self.assertEqual(context.operation, "test_operation")
        self.assertIsNone(context.selection_rect)
        self.assertIsNone(context.image_shape)
        self.assertIsNone(context.mode)
        self.assertEqual(context.retry_count, 0)
        self.assertGreater(context.timestamp, 0)
    
    def test_error_context_with_parameters(self):
        """Test ErrorContext creation with all parameters."""
        selection_rect = (10, 10, 100, 100)
        image_shape = (480, 640)
        
        context = ErrorContext(
            operation="shape_detection",
            selection_rect=selection_rect,
            image_shape=image_shape,
            mode="circle",
            retry_count=2
        )
        
        self.assertEqual(context.operation, "shape_detection")
        self.assertEqual(context.selection_rect, selection_rect)
        self.assertEqual(context.image_shape, image_shape)
        self.assertEqual(context.mode, "circle")
        self.assertEqual(context.retry_count, 2)


class TestManualSelectionValidator(unittest.TestCase):
    """Test ManualSelectionValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ManualSelectionValidator()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.valid_selection = (50, 50, 100, 100)
    
    def test_validate_selection_rect_valid(self):
        """Test validation of valid selection rectangle."""
        # Should not raise any exception
        self.validator.validate_selection_rect(self.valid_selection, self.test_image.shape)
    
    def test_validate_selection_rect_negative_coordinates(self):
        """Test validation fails for negative coordinates."""
        invalid_selection = (-10, 50, 100, 100)
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_selection_rect(invalid_selection, self.test_image.shape)
        
        self.assertIn("negative", str(cm.exception))
    
    def test_validate_selection_rect_zero_dimensions(self):
        """Test validation fails for zero dimensions."""
        invalid_selection = (50, 50, 0, 100)
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_selection_rect(invalid_selection, self.test_image.shape)
        
        self.assertIn("positive", str(cm.exception))
    
    def test_validate_selection_rect_too_small(self):
        """Test validation fails for selections that are too small."""
        small_selection = (50, 50, 10, 10)  # Smaller than minimum
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_selection_rect(small_selection, self.test_image.shape)
        
        self.assertIn("too small", str(cm.exception))
    
    def test_validate_selection_rect_out_of_bounds(self):
        """Test validation fails for out-of-bounds selections."""
        out_of_bounds = (600, 50, 100, 100)  # Extends beyond image width
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_selection_rect(out_of_bounds, self.test_image.shape)
        
        self.assertIn("beyond image bounds", str(cm.exception))
    
    def test_validate_selection_rect_too_large(self):
        """Test validation fails for selections that are too large."""
        large_selection = (0, 0, 640, 480)  # Entire image (> 80% threshold)
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_selection_rect(large_selection, self.test_image.shape)
        
        self.assertIn("too large", str(cm.exception))
    
    def test_validate_image_parameters_valid(self):
        """Test validation of valid image parameters."""
        # Should not raise any exception
        self.validator.validate_image_parameters(self.test_image)
    
    def test_validate_image_parameters_none(self):
        """Test validation fails for None image."""
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_image_parameters(None)
        
        self.assertIn("cannot be None", str(cm.exception))
    
    def test_validate_image_parameters_invalid_shape(self):
        """Test validation fails for invalid image shape."""
        invalid_image = np.array([1, 2, 3])  # 1D array
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_image_parameters(invalid_image)
        
        self.assertIn("Invalid image shape", str(cm.exception))
    
    def test_validate_image_parameters_too_small(self):
        """Test validation fails for images that are too small."""
        small_image = np.ones((10, 10, 3), dtype=np.uint8)
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_image_parameters(small_image)
        
        self.assertIn("too small", str(cm.exception))
    
    def test_validate_image_parameters_empty(self):
        """Test validation fails for empty images."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with self.assertRaises(SelectionValidationError) as cm:
            self.validator.validate_image_parameters(empty_image)
        
        self.assertIn("empty", str(cm.exception))
    
    def test_validate_shape_result_valid(self):
        """Test validation of valid shape result."""
        valid_result = {
            "type": "circle",
            "center": (100, 100),
            "dimensions": (50, 50),
            "confidence_score": 0.8
        }
        
        # Should not raise any exception
        self.validator.validate_shape_result(valid_result)
    
    def test_validate_shape_result_missing_field(self):
        """Test validation fails for missing required fields."""
        invalid_result = {
            "type": "circle",
            "center": (100, 100)
            # Missing dimensions and confidence_score
        }
        
        with self.assertRaises(ShapeDetectionError) as cm:
            self.validator.validate_shape_result(invalid_result)
        
        self.assertIn("Missing required field", str(cm.exception))
    
    def test_validate_shape_result_invalid_type(self):
        """Test validation fails for invalid shape type."""
        invalid_result = {
            "type": "triangle",  # Invalid type
            "center": (100, 100),
            "dimensions": (50, 50),
            "confidence_score": 0.8
        }
        
        with self.assertRaises(ShapeDetectionError) as cm:
            self.validator.validate_shape_result(invalid_result)
        
        self.assertIn("Invalid shape type", str(cm.exception))
    
    def test_validate_shape_result_low_confidence(self):
        """Test validation fails for low confidence scores."""
        low_confidence_result = {
            "type": "circle",
            "center": (100, 100),
            "dimensions": (50, 50),
            "confidence_score": 0.3  # Below threshold
        }
        
        with self.assertRaises(ShapeDetectionError) as cm:
            self.validator.validate_shape_result(low_confidence_result)
        
        self.assertIn("confidence too low", str(cm.exception))
    
    @patch('config.ENABLE_SELECTION_VALIDATION', False)
    def test_validation_disabled(self):
        """Test that validation can be disabled via configuration."""
        validator = ManualSelectionValidator()
        
        # These should not raise exceptions when validation is disabled
        validator.validate_selection_rect((-10, -10, 5, 5), (100, 100))
        validator.validate_image_parameters(None)


class TestErrorRecoveryManager(unittest.TestCase):
    """Test ErrorRecoveryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recovery_manager = ErrorRecoveryManager()
        self.test_context = ErrorContext(
            operation="test_operation",
            selection_rect=(50, 50, 100, 100),
            image_shape=(480, 640),
            mode="circle"
        )
    
    def test_handle_selection_validation_error(self):
        """Test handling of selection validation errors."""
        error = SelectionValidationError("Selection too small", (50, 50, 10, 10))
        
        result = self.recovery_manager.handle_selection_error(error, self.test_context)
        
        self.assertIsNotNone(result)
        self.assertIn("adjusted_selection", result)
        self.assertEqual(self.recovery_manager.recovery_stats["total_errors"], 1)
    
    def test_handle_shape_detection_error(self):
        """Test handling of shape detection errors."""
        error = ShapeDetectionError("No shape found", "circle")
        
        result = self.recovery_manager.handle_selection_error(error, self.test_context)
        
        self.assertIsNotNone(result)
        self.assertIn("suggested_selection", result)
        self.assertEqual(result["recovery_method"], "area_expansion")
    
    def test_handle_enhanced_analysis_error(self):
        """Test handling of enhanced analysis errors."""
        error = EnhancedAnalysisError("Analysis failed", fallback_available=True)
        
        result = self.recovery_manager.handle_selection_error(error, self.test_context)
        
        self.assertIsNotNone(result)
        self.assertTrue(result["use_fallback"])
        self.assertEqual(result["recovery_method"], "standard_analysis_fallback")
        self.assertEqual(self.recovery_manager.recovery_stats["fallback_uses"], 1)
    
    def test_handle_timeout_error(self):
        """Test handling of timeout errors."""
        error = TimeoutError("Operation timed out", 5000)
        
        result = self.recovery_manager.handle_selection_error(error, self.test_context)
        
        self.assertIsNotNone(result)
        self.assertIn("reduced_selection", result)
        self.assertEqual(result["recovery_method"], "size_reduction")
    
    def test_should_retry_recoverable_error(self):
        """Test retry decision for recoverable errors."""
        error = SelectionValidationError("Test error")
        context = ErrorContext("test", retry_count=1)
        
        should_retry = self.recovery_manager.should_retry(error, context)
        
        self.assertTrue(should_retry)
    
    def test_should_retry_max_retries_exceeded(self):
        """Test retry decision when max retries exceeded."""
        error = SelectionValidationError("Test error")
        context = ErrorContext("test", retry_count=5)  # Exceeds max retries
        
        should_retry = self.recovery_manager.should_retry(error, context)
        
        self.assertFalse(should_retry)
    
    def test_should_retry_non_recoverable_error(self):
        """Test retry decision for non-recoverable errors."""
        error = ConfigurationError("Config error")  # Non-recoverable
        context = ErrorContext("test", retry_count=0)
        
        should_retry = self.recovery_manager.should_retry(error, context)
        
        self.assertFalse(should_retry)
    
    def test_get_user_message_with_retry_count(self):
        """Test user message generation with retry count."""
        error = SelectionValidationError("Test error")
        context = ErrorContext("test", retry_count=2)
        
        message = self.recovery_manager.get_user_message(error, context)
        
        self.assertIn("Attempt 3", message)
        self.assertIn("Suggestions:", message)
    
    def test_get_recovery_stats(self):
        """Test recovery statistics calculation."""
        # Simulate some errors
        error1 = SelectionValidationError("Error 1")
        error2 = ShapeDetectionError("Error 2")
        
        self.recovery_manager.handle_selection_error(error1, self.test_context)
        self.recovery_manager.handle_selection_error(error2, self.test_context)
        
        stats = self.recovery_manager.get_recovery_stats()
        
        self.assertEqual(stats["total_errors"], 2)
        self.assertEqual(stats["recovered_errors"], 2)
        self.assertEqual(stats["recovery_rate"], 1.0)
    
    def test_reset_stats(self):
        """Test resetting recovery statistics."""
        # Add some errors first
        error = SelectionValidationError("Test error")
        self.recovery_manager.handle_selection_error(error, self.test_context)
        
        # Reset stats
        self.recovery_manager.reset_stats()
        
        stats = self.recovery_manager.get_recovery_stats()
        self.assertEqual(stats["total_errors"], 0)
        self.assertEqual(stats["recovered_errors"], 0)
        self.assertEqual(len(self.recovery_manager.error_history), 0)


class TestUserFeedbackManager(unittest.TestCase):
    """Test UserFeedbackManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feedback_manager = UserFeedbackManager()
        self.test_error = SelectionValidationError("Test error")
        self.test_context = ErrorContext("test_operation")
    
    def test_add_error_message(self):
        """Test adding error messages."""
        self.feedback_manager.add_error_message(
            self.test_error, self.test_context, ErrorSeverity.HIGH
        )
        
        message = self.feedback_manager.get_current_message()
        self.assertIsNotNone(message)
        self.assertEqual(message, self.test_error.user_message)
        self.assertEqual(self.feedback_manager.get_current_message_type(), "error")
    
    def test_add_info_message(self):
        """Test adding informational messages."""
        self.feedback_manager.add_info_message("Test info", "test_context")
        
        message = self.feedback_manager.get_current_message()
        self.assertEqual(message, "Test info")
        self.assertEqual(self.feedback_manager.get_current_message_type(), "info")
    
    def test_add_success_message(self):
        """Test adding success messages."""
        self.feedback_manager.add_success_message("Test success", "test_context")
        
        message = self.feedback_manager.get_current_message()
        self.assertEqual(message, "Test success")
        self.assertEqual(self.feedback_manager.get_current_message_type(), "success")
    
    def test_message_priority(self):
        """Test message prioritization by severity."""
        # Add low priority message first
        self.feedback_manager.add_info_message("Low priority")
        
        # Add high priority message
        self.feedback_manager.add_error_message(
            self.test_error, self.test_context, ErrorSeverity.CRITICAL
        )
        
        # High priority message should be current
        message = self.feedback_manager.get_current_message()
        self.assertEqual(message, self.test_error.user_message)
    
    def test_message_timeout(self):
        """Test message timeout functionality."""
        # Mock time to control timeout
        with patch('time.time') as mock_time:
            mock_time.return_value = 0.0
            
            self.feedback_manager.add_info_message("Test message")
            self.assertIsNotNone(self.feedback_manager.get_current_message())
            
            # Simulate time passing beyond timeout
            mock_time.return_value = 4.0  # 4 seconds = 4000ms > 3000ms timeout
            
            message = self.feedback_manager.get_current_message()
            self.assertIsNone(message)
    
    def test_clear_messages(self):
        """Test clearing all messages."""
        self.feedback_manager.add_error_message(self.test_error, self.test_context)
        self.feedback_manager.add_info_message("Test info")
        
        self.feedback_manager.clear_messages()
        
        self.assertIsNone(self.feedback_manager.get_current_message())
        self.assertEqual(len(self.feedback_manager.message_queue), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for error handling."""
    
    def test_create_error_context(self):
        """Test error context creation utility."""
        context = create_error_context(
            "test_operation",
            selection_rect=(10, 10, 100, 100),
            mode="circle"
        )
        
        self.assertEqual(context.operation, "test_operation")
        self.assertEqual(context.selection_rect, (10, 10, 100, 100))
        self.assertEqual(context.mode, "circle")
    
    def test_validate_manual_selection_operation_valid(self):
        """Test validation of valid manual selection operation."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        selection_rect = (50, 50, 100, 100)
        
        # Should not raise any exception
        validate_manual_selection_operation(image, selection_rect)
    
    def test_validate_manual_selection_operation_invalid(self):
        """Test validation of invalid manual selection operation."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        invalid_selection = (-10, 50, 100, 100)  # Negative coordinates
        
        with self.assertRaises(ManualSelectionError):
            validate_manual_selection_operation(image, invalid_selection)
    
    def test_handle_operation_with_recovery_success(self):
        """Test successful operation execution with recovery handler."""
        def successful_operation():
            return "success"
        
        recovery_manager = ErrorRecoveryManager()
        context = ErrorContext("test_operation")
        
        result = handle_operation_with_recovery(
            successful_operation, recovery_manager, context
        )
        
        self.assertEqual(result, "success")
    
    def test_handle_operation_with_recovery_with_retries(self):
        """Test operation execution with retries."""
        call_count = 0
        
        def failing_then_succeeding_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise SelectionValidationError("Temporary failure")
            return "success"
        
        recovery_manager = ErrorRecoveryManager()
        context = ErrorContext("test_operation")
        
        result = handle_operation_with_recovery(
            failing_then_succeeding_operation, recovery_manager, context
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
        self.assertGreater(recovery_manager.recovery_stats["retry_attempts"], 0)
    
    def test_handle_operation_with_recovery_max_retries(self):
        """Test operation execution that exceeds max retries."""
        def always_failing_operation():
            raise SelectionValidationError("Always fails")
        
        recovery_manager = ErrorRecoveryManager()
        context = ErrorContext("test_operation")
        
        with self.assertRaises(ManualSelectionError) as cm:
            handle_operation_with_recovery(
                always_failing_operation, recovery_manager, context
            )
        
        self.assertIn("Maximum retry attempts", str(cm.exception))


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration with configuration parameters."""
    
    def test_configuration_parameter_usage(self):
        """Test that configuration parameters are properly used."""
        # Test that validator uses config parameters
        validator = ManualSelectionValidator()
        
        self.assertEqual(validator.min_selection_size, 
                        getattr(config, 'MIN_SELECTION_SIZE_PX', 20))
        self.assertEqual(validator.max_area_ratio, 
                        getattr(config, 'MAX_SELECTION_AREA_RATIO', 0.8))
    
    def test_error_recovery_configuration(self):
        """Test that error recovery uses config parameters."""
        recovery_manager = ErrorRecoveryManager()
        
        self.assertEqual(recovery_manager.max_retries, 
                        getattr(config, 'MAX_SELECTION_RETRIES', 3))
        self.assertEqual(recovery_manager.enable_fallback, 
                        getattr(config, 'FALLBACK_TO_STANDARD_THRESHOLD', True))
    
    def test_feedback_manager_configuration(self):
        """Test that feedback manager uses config parameters."""
        feedback_manager = UserFeedbackManager()
        
        self.assertEqual(feedback_manager.display_time_ms, 
                        getattr(config, 'ERROR_MESSAGE_DISPLAY_TIME_MS', 3000))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)