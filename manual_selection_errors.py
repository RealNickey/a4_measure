"""
Error Handling and Validation for Manual Shape Selection

This module provides comprehensive error handling, validation, and fallback strategies
for manual shape selection operations. It includes custom exception classes, error
recovery mechanisms, and user feedback message generation.

Requirements addressed: 4.3, 2.3
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import config


class ManualSelectionError(Exception):
    """Base exception class for manual selection errors."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN", 
                 recoverable: bool = True, user_message: Optional[str] = None):
        """
        Initialize manual selection error.
        
        Args:
            message: Technical error message
            error_code: Error code for categorization
            recoverable: Whether the error can be recovered from
            user_message: User-friendly error message
        """
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable
        self.user_message = user_message or message
        self.timestamp = time.time()


class SelectionValidationError(ManualSelectionError):
    """Error raised when selection validation fails."""
    
    def __init__(self, message: str, selection_rect: Optional[Tuple[int, int, int, int]] = None):
        super().__init__(
            message, 
            error_code="SELECTION_INVALID",
            recoverable=True,
            user_message="Selection area is invalid. Please try selecting again."
        )
        self.selection_rect = selection_rect


class ShapeDetectionError(ManualSelectionError):
    """Error raised when shape detection fails."""
    
    def __init__(self, message: str, shape_type: Optional[str] = None):
        super().__init__(
            message,
            error_code="SHAPE_NOT_FOUND",
            recoverable=True,
            user_message=f"No {shape_type or 'shape'} detected in selection. Try a different area."
        )
        self.shape_type = shape_type


class EnhancedAnalysisError(ManualSelectionError):
    """Error raised when enhanced contour analysis fails."""
    
    def __init__(self, message: str, fallback_available: bool = True):
        super().__init__(
            message,
            error_code="ENHANCED_ANALYSIS_FAILED",
            recoverable=fallback_available,
            user_message="Enhanced analysis failed. Using standard detection."
        )
        self.fallback_available = fallback_available


class ConfigurationError(ManualSelectionError):
    """Error raised when configuration parameters are invalid."""
    
    def __init__(self, message: str, parameter_name: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIG_INVALID",
            recoverable=False,
            user_message="Configuration error. Please check system settings."
        )
        self.parameter_name = parameter_name


class TimeoutError(ManualSelectionError):
    """Error raised when operations timeout."""
    
    def __init__(self, message: str, timeout_ms: int):
        super().__init__(
            message,
            error_code="OPERATION_TIMEOUT",
            recoverable=True,
            user_message=f"Operation timed out after {timeout_ms}ms. Please try again."
        )
        self.timeout_ms = timeout_ms


class ErrorSeverity(Enum):
    """Severity levels for error classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    selection_rect: Optional[Tuple[int, int, int, int]] = None
    image_shape: Optional[Tuple[int, int]] = None
    mode: Optional[str] = None
    retry_count: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ManualSelectionValidator:
    """
    Comprehensive validator for manual selection operations.
    
    Provides validation for selection rectangles, image parameters,
    configuration settings, and operation contexts.
    """
    
    def __init__(self):
        """Initialize the validator with configuration parameters."""
        self.min_selection_size = getattr(config, 'MIN_SELECTION_SIZE_PX', 20) or 20
        self.max_area_ratio = getattr(config, 'MAX_SELECTION_AREA_RATIO', 0.8) or 0.8
        self.snap_distance = getattr(config, 'SELECTION_SNAP_DISTANCE_PX', 15) or 15
        self.enable_validation = getattr(config, 'ENABLE_SELECTION_VALIDATION', True)
    
    def validate_selection_rect(self, selection_rect: Tuple[int, int, int, int],
                               image_shape: Tuple[int, int]) -> None:
        """
        Validate selection rectangle parameters.
        
        Args:
            selection_rect: Selection rectangle as (x, y, width, height)
            image_shape: Image shape as (height, width)
            
        Raises:
            SelectionValidationError: If validation fails
        """
        if not self.enable_validation:
            return
        
        x, y, w, h = selection_rect
        img_h, img_w = image_shape[:2]
        
        # Check for negative coordinates
        if x < 0 or y < 0:
            raise SelectionValidationError(
                f"Selection coordinates cannot be negative: ({x}, {y})",
                selection_rect
            )
        
        # Check for zero or negative dimensions
        if w <= 0 or h <= 0:
            raise SelectionValidationError(
                f"Selection dimensions must be positive: {w}x{h}",
                selection_rect
            )
        
        # Check minimum size
        if w < self.min_selection_size or h < self.min_selection_size:
            raise SelectionValidationError(
                f"Selection too small: {w}x{h}. Minimum size: {self.min_selection_size}px",
                selection_rect
            )
        
        # Check bounds
        if x + w > img_w or y + h > img_h:
            raise SelectionValidationError(
                f"Selection extends beyond image bounds: ({x}, {y}, {w}, {h}) vs {img_w}x{img_h}",
                selection_rect
            )
        
        # Check maximum area (prevent extremely large selections)
        selection_area = w * h
        image_area = img_w * img_h
        if selection_area > image_area * self.max_area_ratio:
            raise SelectionValidationError(
                f"Selection area too large: {selection_area} > {image_area * self.max_area_ratio}",
                selection_rect
            )
    
    def validate_image_parameters(self, image: np.ndarray) -> None:
        """
        Validate image parameters for processing.
        
        Args:
            image: Input image to validate
            
        Raises:
            SelectionValidationError: If image is invalid
        """
        if not self.enable_validation:
            return
        
        if image is None:
            raise SelectionValidationError("Image cannot be None")
        
        if len(image.shape) < 2:
            raise SelectionValidationError(f"Invalid image shape: {image.shape}")
        
        if image.shape[0] < self.min_selection_size or image.shape[1] < self.min_selection_size:
            raise SelectionValidationError(
                f"Image too small for selection: {image.shape[:2]}. "
                f"Minimum: {self.min_selection_size}x{self.min_selection_size}"
            )
        
        # Check for empty image
        if np.all(image == 0):
            raise SelectionValidationError("Image appears to be empty (all zeros)")
    
    def validate_shape_result(self, shape_result: Dict[str, Any]) -> None:
        """
        Validate shape detection result.
        
        Args:
            shape_result: Shape detection result to validate
            
        Raises:
            ShapeDetectionError: If result is invalid
        """
        if not self.enable_validation:
            return
        
        required_fields = ["type", "center", "dimensions", "confidence_score"]
        
        for field in required_fields:
            if field not in shape_result:
                raise ShapeDetectionError(f"Missing required field: {field}")
        
        # Validate shape type
        if shape_result["type"] not in ["circle", "rectangle"]:
            raise ShapeDetectionError(f"Invalid shape type: {shape_result['type']}")
        
        # Validate confidence score
        confidence = shape_result["confidence_score"]
        if not (0.0 <= confidence <= 1.0):
            raise ShapeDetectionError(f"Invalid confidence score: {confidence}")
        
        # Check minimum confidence threshold
        min_confidence = getattr(config, 'SHAPE_CONFIDENCE_THRESHOLD', 0.5)
        if confidence < min_confidence:
            raise ShapeDetectionError(
                f"Shape confidence too low: {confidence:.2f} < {min_confidence:.2f}",
                shape_result["type"]
            )
    
    def validate_configuration(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check minimum selection size
        if self.min_selection_size <= 0:
            raise ConfigurationError(
                f"MIN_SELECTION_SIZE_PX must be positive: {self.min_selection_size}",
                "MIN_SELECTION_SIZE_PX"
            )
        
        # Check area ratio
        if not (0.0 < self.max_area_ratio <= 1.0):
            raise ConfigurationError(
                f"MAX_SELECTION_AREA_RATIO must be between 0 and 1: {self.max_area_ratio}",
                "MAX_SELECTION_AREA_RATIO"
            )
        
        # Check snap distance
        if self.snap_distance < 0:
            raise ConfigurationError(
                f"SELECTION_SNAP_DISTANCE_PX cannot be negative: {self.snap_distance}",
                "SELECTION_SNAP_DISTANCE_PX"
            )


class ErrorRecoveryManager:
    """
    Manages error recovery and fallback strategies for manual selection operations.
    
    Provides automatic retry mechanisms, fallback strategies, and error context tracking.
    """
    
    def __init__(self):
        """Initialize the error recovery manager."""
        self.max_retries = getattr(config, 'MAX_SELECTION_RETRIES', 3)
        self.enable_fallback = getattr(config, 'FALLBACK_TO_STANDARD_THRESHOLD', True)
        self.min_selection_size = getattr(config, 'MIN_SELECTION_SIZE_PX', 20)
        self.error_history: List[ManualSelectionError] = []
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "fallback_uses": 0,
            "retry_attempts": 0
        }
    
    def handle_selection_error(self, error: ManualSelectionError, 
                             context: ErrorContext) -> Optional[Dict[str, Any]]:
        """
        Handle a manual selection error with appropriate recovery strategy.
        
        Args:
            error: The error that occurred
            context: Context information for the error
            
        Returns:
            Recovery result or None if recovery failed
        """
        self.error_history.append(error)
        self.recovery_stats["total_errors"] += 1
        
        # Log error for debugging
        self._log_error(error, context)
        
        # Attempt recovery based on error type
        recovery_result = None
        
        if isinstance(error, SelectionValidationError):
            recovery_result = self._recover_from_validation_error(error, context)
        elif isinstance(error, ShapeDetectionError):
            recovery_result = self._recover_from_detection_error(error, context)
        elif isinstance(error, EnhancedAnalysisError):
            recovery_result = self._recover_from_analysis_error(error, context)
        elif isinstance(error, TimeoutError):
            recovery_result = self._recover_from_timeout_error(error, context)
        
        if recovery_result is not None:
            self.recovery_stats["recovered_errors"] += 1
        
        return recovery_result
    
    def should_retry(self, error: ManualSelectionError, context: ErrorContext) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error: The error that occurred
            context: Context information
            
        Returns:
            True if retry should be attempted
        """
        if not error.recoverable:
            return False
        
        if context.retry_count >= self.max_retries:
            return False
        
        # Don't retry configuration errors
        if isinstance(error, ConfigurationError):
            return False
        
        return True
    
    def get_user_message(self, error: ManualSelectionError, 
                        context: ErrorContext) -> str:
        """
        Generate user-friendly error message.
        
        Args:
            error: The error that occurred
            context: Context information
            
        Returns:
            User-friendly error message
        """
        base_message = error.user_message
        
        # Add context-specific information
        if context.retry_count > 0:
            base_message += f" (Attempt {context.retry_count + 1}/{self.max_retries + 1})"
        
        # Add recovery suggestions
        suggestions = self._get_recovery_suggestions(error, context)
        if suggestions:
            base_message += f" Suggestions: {', '.join(suggestions)}"
        
        return base_message
    
    def _recover_from_validation_error(self, error: SelectionValidationError,
                                     context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt recovery from selection validation error."""
        if context.selection_rect is None:
            return None
        
        x, y, w, h = context.selection_rect
        
        # Try to adjust selection to minimum size
        if w < self.min_selection_size or h < self.min_selection_size:
            new_w = max(w, self.min_selection_size)
            new_h = max(h, self.min_selection_size)
            
            # Adjust position to keep within bounds if needed
            if context.image_shape:
                img_h, img_w = context.image_shape[:2]
                if x + new_w > img_w:
                    x = max(0, img_w - new_w)
                if y + new_h > img_h:
                    y = max(0, img_h - new_h)
            
            return {
                "adjusted_selection": (x, y, new_w, new_h),
                "recovery_method": "size_adjustment"
            }
        
        return {
            "recovery_method": "validation_attempted",
            "message": "Validation error handled"
        }
    
    def _recover_from_detection_error(self, error: ShapeDetectionError,
                                    context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt recovery from shape detection error."""
        # Suggest expanding selection area
        if context.selection_rect and context.image_shape:
            x, y, w, h = context.selection_rect
            img_h, img_w = context.image_shape[:2]
            
            # Expand by 20% in each direction
            expand_x = int(w * 0.1)
            expand_y = int(h * 0.1)
            
            new_x = max(0, x - expand_x)
            new_y = max(0, y - expand_y)
            new_w = min(w + 2 * expand_x, img_w - new_x)
            new_h = min(h + 2 * expand_y, img_h - new_y)
            
            return {
                "suggested_selection": (new_x, new_y, new_w, new_h),
                "recovery_method": "area_expansion"
            }
        
        return {
            "recovery_method": "detection_attempted",
            "message": "Shape detection error handled"
        }
    
    def _recover_from_analysis_error(self, error: EnhancedAnalysisError,
                                   context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt recovery from enhanced analysis error."""
        if self.enable_fallback and error.fallback_available:
            self.recovery_stats["fallback_uses"] += 1
            return {
                "use_fallback": True,
                "recovery_method": "standard_analysis_fallback"
            }
        
        return None
    
    def _recover_from_timeout_error(self, error: TimeoutError,
                                  context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt recovery from timeout error."""
        # Suggest reducing selection size for faster processing
        if context.selection_rect:
            x, y, w, h = context.selection_rect
            
            # Reduce by 25%
            new_w = int(w * 0.75)
            new_h = int(h * 0.75)
            new_x = x + (w - new_w) // 2
            new_y = y + (h - new_h) // 2
            
            return {
                "reduced_selection": (new_x, new_y, new_w, new_h),
                "recovery_method": "size_reduction"
            }
        
        return None
    
    def _get_recovery_suggestions(self, error: ManualSelectionError,
                                context: ErrorContext) -> List[str]:
        """Get recovery suggestions for the user."""
        suggestions = []
        
        if isinstance(error, SelectionValidationError):
            suggestions.extend([
                "Try selecting a larger area",
                "Ensure selection is within image bounds"
            ])
        elif isinstance(error, ShapeDetectionError):
            suggestions.extend([
                "Try selecting a different area",
                "Ensure the shape is clearly visible",
                "Check lighting conditions"
            ])
        elif isinstance(error, EnhancedAnalysisError):
            suggestions.append("Standard analysis will be used instead")
        elif isinstance(error, TimeoutError):
            suggestions.extend([
                "Try selecting a smaller area",
                "Ensure good system performance"
            ])
        
        return suggestions
    
    def _log_error(self, error: ManualSelectionError, context: ErrorContext) -> None:
        """Log error information for debugging."""
        # In a production system, this would log to a proper logging system
        print(f"Manual Selection Error: {error.error_code} - {error}")
        print(f"Context: {context.operation}, Retry: {context.retry_count}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        stats = self.recovery_stats.copy()
        if stats["total_errors"] > 0:
            stats["recovery_rate"] = stats["recovered_errors"] / stats["total_errors"]
        else:
            stats["recovery_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset recovery statistics."""
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "fallback_uses": 0,
            "retry_attempts": 0
        }
        self.error_history.clear()


class UserFeedbackManager:
    """
    Manages user feedback messages for manual selection operations.
    
    Provides categorized feedback messages, display timing, and message prioritization.
    """
    
    def __init__(self):
        """Initialize the user feedback manager."""
        self.display_time_ms = getattr(config, 'ERROR_MESSAGE_DISPLAY_TIME_MS', 3000)
        self.message_queue: List[Dict[str, Any]] = []
        self.current_message: Optional[Dict[str, Any]] = None
        self.message_start_time = 0.0
    
    def add_error_message(self, error: ManualSelectionError, 
                         context: ErrorContext,
                         severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> None:
        """
        Add an error message to the feedback queue.
        
        Args:
            error: The error that occurred
            context: Context information
            severity: Message severity level
        """
        message = {
            "text": error.user_message,
            "type": "error",
            "severity": severity,
            "error_code": error.error_code,
            "timestamp": time.time(),
            "context": context.operation
        }
        
        self.message_queue.append(message)
        self._update_current_message()
    
    def add_info_message(self, text: str, context: str = "general") -> None:
        """
        Add an informational message.
        
        Args:
            text: Message text
            context: Context information
        """
        message = {
            "text": text,
            "type": "info",
            "severity": ErrorSeverity.LOW,
            "timestamp": time.time(),
            "context": context
        }
        
        self.message_queue.append(message)
        self._update_current_message()
    
    def add_success_message(self, text: str, context: str = "general") -> None:
        """
        Add a success message.
        
        Args:
            text: Message text
            context: Context information
        """
        message = {
            "text": text,
            "type": "success",
            "severity": ErrorSeverity.LOW,
            "timestamp": time.time(),
            "context": context
        }
        
        self.message_queue.append(message)
        self._update_current_message()
    
    def get_current_message(self) -> Optional[str]:
        """
        Get the current message to display.
        
        Returns:
            Current message text or None if no message should be displayed
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        
        if (self.current_message and 
            current_time - self.message_start_time > self.display_time_ms):
            self.current_message = None
            self._update_current_message()
        
        return self.current_message["text"] if self.current_message else None
    
    def get_current_message_type(self) -> Optional[str]:
        """
        Get the type of the current message.
        
        Returns:
            Message type ("error", "info", "success") or None
        """
        return self.current_message["type"] if self.current_message else None
    
    def clear_messages(self) -> None:
        """Clear all messages from the queue."""
        self.message_queue.clear()
        self.current_message = None
        self.message_start_time = 0.0
    
    def _update_current_message(self) -> None:
        """Update the current message based on priority."""
        if not self.message_queue:
            return
        
        # Sort by severity (highest first) and timestamp (newest first)
        severity_order = {
            ErrorSeverity.CRITICAL: 4,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.LOW: 1
        }
        
        self.message_queue.sort(
            key=lambda m: (severity_order[m["severity"]], -m["timestamp"]),
            reverse=True
        )
        
        # Set the highest priority message as current
        if not self.current_message or self.message_queue[0]["severity"] != self.current_message.get("severity"):
            self.current_message = self.message_queue.pop(0)
            self.message_start_time = time.time() * 1000


# Utility functions for error handling

def create_error_context(operation: str, **kwargs) -> ErrorContext:
    """
    Create an error context with the given parameters.
    
    Args:
        operation: Name of the operation
        **kwargs: Additional context parameters
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(operation=operation, **kwargs)


def validate_manual_selection_operation(image: np.ndarray,
                                      selection_rect: Tuple[int, int, int, int],
                                      operation: str = "manual_selection") -> None:
    """
    Validate a complete manual selection operation.
    
    Args:
        image: Input image
        selection_rect: Selection rectangle
        operation: Operation name for context
        
    Raises:
        ManualSelectionError: If validation fails
    """
    validator = ManualSelectionValidator()
    
    try:
        validator.validate_configuration()
        validator.validate_image_parameters(image)
        validator.validate_selection_rect(selection_rect, image.shape)
    except Exception as e:
        if isinstance(e, ManualSelectionError):
            raise
        else:
            raise ManualSelectionError(f"Validation failed: {str(e)}", "VALIDATION_ERROR")


def handle_operation_with_recovery(operation_func, error_manager: ErrorRecoveryManager,
                                 context: ErrorContext, *args, **kwargs) -> Any:
    """
    Execute an operation with automatic error recovery.
    
    Args:
        operation_func: Function to execute
        error_manager: Error recovery manager
        context: Error context
        *args, **kwargs: Arguments for the operation function
        
    Returns:
        Operation result or None if all recovery attempts failed
    """
    while context.retry_count <= error_manager.max_retries:
        try:
            return operation_func(*args, **kwargs)
        except ManualSelectionError as e:
            recovery_result = error_manager.handle_selection_error(e, context)
            
            if not error_manager.should_retry(e, context):
                raise
            
            context.retry_count += 1
            error_manager.recovery_stats["retry_attempts"] += 1
            
            # Apply recovery adjustments if available
            if recovery_result and "adjusted_selection" in recovery_result:
                # Update selection rectangle in kwargs if present
                if "selection_rect" in kwargs:
                    kwargs["selection_rect"] = recovery_result["adjusted_selection"]
    
    raise ManualSelectionError("Maximum retry attempts exceeded", "MAX_RETRIES_EXCEEDED")