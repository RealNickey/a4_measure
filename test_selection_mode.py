"""
Unit tests for selection mode management system.

Tests the SelectionMode enum and ModeManager class functionality including
mode transitions, state consistency, and indicator methods.
"""

import unittest
from selection_mode import SelectionMode, ModeManager


class TestSelectionMode(unittest.TestCase):
    """Test cases for SelectionMode enum."""
    
    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(SelectionMode.AUTO.value, "auto")
        self.assertEqual(SelectionMode.MANUAL_RECTANGLE.value, "manual_rect")
        self.assertEqual(SelectionMode.MANUAL_CIRCLE.value, "manual_circle")
    
    def test_string_representation(self):
        """Test string representation of enum values."""
        self.assertEqual(str(SelectionMode.AUTO), "auto")
        self.assertEqual(str(SelectionMode.MANUAL_RECTANGLE), "manual_rect")
        self.assertEqual(str(SelectionMode.MANUAL_CIRCLE), "manual_circle")


class TestModeManager(unittest.TestCase):
    """Test cases for ModeManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mode_manager = ModeManager()
    
    def test_initial_mode_default(self):
        """Test that default initial mode is AUTO."""
        manager = ModeManager()
        self.assertEqual(manager.get_current_mode(), SelectionMode.AUTO)
    
    def test_initial_mode_custom(self):
        """Test initialization with custom initial mode."""
        manager = ModeManager(SelectionMode.MANUAL_RECTANGLE)
        self.assertEqual(manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
    
    def test_mode_cycling_sequence(self):
        """Test that mode cycling follows the correct sequence."""
        # Start with AUTO
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.AUTO)
        
        # Cycle to MANUAL_RECTANGLE
        next_mode = self.mode_manager.cycle_mode()
        self.assertEqual(next_mode, SelectionMode.MANUAL_RECTANGLE)
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.MANUAL_RECTANGLE)
        
        # Cycle to MANUAL_CIRCLE
        next_mode = self.mode_manager.cycle_mode()
        self.assertEqual(next_mode, SelectionMode.MANUAL_CIRCLE)
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.MANUAL_CIRCLE)
        
        # Cycle back to AUTO
        next_mode = self.mode_manager.cycle_mode()
        self.assertEqual(next_mode, SelectionMode.AUTO)
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.AUTO)
    
    def test_multiple_cycles(self):
        """Test multiple complete cycles maintain consistency."""
        initial_mode = self.mode_manager.get_current_mode()
        
        # Complete two full cycles
        for _ in range(6):  # 2 cycles * 3 modes each
            self.mode_manager.cycle_mode()
        
        # Should be back to initial mode
        self.assertEqual(self.mode_manager.get_current_mode(), initial_mode)
    
    def test_set_mode_valid(self):
        """Test setting mode to valid values."""
        self.mode_manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.MANUAL_CIRCLE)
        
        self.mode_manager.set_mode(SelectionMode.AUTO)
        self.assertEqual(self.mode_manager.get_current_mode(), SelectionMode.AUTO)
    
    def test_set_mode_invalid(self):
        """Test that setting invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            self.mode_manager.set_mode("invalid_mode")
        
        with self.assertRaises(ValueError):
            self.mode_manager.set_mode(123)
    
    def test_mode_indicators(self):
        """Test mode indicator strings."""
        self.mode_manager.set_mode(SelectionMode.AUTO)
        self.assertEqual(self.mode_manager.get_mode_indicator(), "AUTO")
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertEqual(self.mode_manager.get_mode_indicator(), "MANUAL RECT")
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertEqual(self.mode_manager.get_mode_indicator(), "MANUAL CIRCLE")
    
    def test_is_manual_mode(self):
        """Test manual mode detection."""
        self.mode_manager.set_mode(SelectionMode.AUTO)
        self.assertFalse(self.mode_manager.is_manual_mode())
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertTrue(self.mode_manager.is_manual_mode())
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertTrue(self.mode_manager.is_manual_mode())
    
    def test_is_auto_mode(self):
        """Test auto mode detection."""
        self.mode_manager.set_mode(SelectionMode.AUTO)
        self.assertTrue(self.mode_manager.is_auto_mode())
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertFalse(self.mode_manager.is_auto_mode())
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertFalse(self.mode_manager.is_auto_mode())
    
    def test_get_manual_shape_type(self):
        """Test manual shape type detection."""
        self.mode_manager.set_mode(SelectionMode.AUTO)
        self.assertIsNone(self.mode_manager.get_manual_shape_type())
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_RECTANGLE)
        self.assertEqual(self.mode_manager.get_manual_shape_type(), "rectangle")
        
        self.mode_manager.set_mode(SelectionMode.MANUAL_CIRCLE)
        self.assertEqual(self.mode_manager.get_manual_shape_type(), "circle")
    
    def test_state_consistency_after_operations(self):
        """Test that state remains consistent after various operations."""
        # Perform various operations
        self.mode_manager.cycle_mode()
        current_mode = self.mode_manager.get_current_mode()
        
        # Check all methods return consistent results
        self.assertEqual(self.mode_manager.get_current_mode(), current_mode)
        
        if current_mode == SelectionMode.AUTO:
            self.assertTrue(self.mode_manager.is_auto_mode())
            self.assertFalse(self.mode_manager.is_manual_mode())
            self.assertIsNone(self.mode_manager.get_manual_shape_type())
        else:
            self.assertFalse(self.mode_manager.is_auto_mode())
            self.assertTrue(self.mode_manager.is_manual_mode())
            self.assertIsNotNone(self.mode_manager.get_manual_shape_type())
    
    def test_mode_cycle_order_consistency(self):
        """Test that the mode cycle order is maintained correctly."""
        expected_order = [
            SelectionMode.AUTO,
            SelectionMode.MANUAL_RECTANGLE,
            SelectionMode.MANUAL_CIRCLE
        ]
        
        # Start from AUTO and cycle through all modes
        self.mode_manager.set_mode(SelectionMode.AUTO)
        actual_order = []
        
        for _ in range(len(expected_order)):
            actual_order.append(self.mode_manager.get_current_mode())
            self.mode_manager.cycle_mode()
        
        self.assertEqual(actual_order, expected_order)


if __name__ == '__main__':
    unittest.main()