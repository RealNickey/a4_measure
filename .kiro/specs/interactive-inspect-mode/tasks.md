# Implementation Plan

- [x] 1. Enhance shape data structure for interactive functionality





  - Modify the shape classification functions to return hit testing polygons alongside measurements
  - Create helper functions to generate hit testing contours for circles and rectangles
  - Add area calculation and inner shape flagging to support selection priority
  - _Requirements: 4.1, 4.4_

- [x] 2. Implement coordinate transformation utilities





  - Write functions to convert between display window coordinates and original image coordinates
  - Create boundary checking utilities to ensure mouse coordinates are within valid ranges
  - Add scaling factor calculation based on display window size
  - _Requirements: 4.2_

- [x] 3. Create hit testing engine with snapping behavior





  - Implement precise hit testing using cv2.pointPolygonTest for shape containment
  - Add proximity-based snapping for shapes near the cursor but not directly under it
  - Create selection priority logic that favors smaller shapes when multiple overlap
  - Write unit tests for hit testing accuracy with various shape configurations
  - _Requirements: 2.2, 2.4, 4.1, 4.3_

- [x] 4. Build interaction state management system





  - Create data structures to track current hover and selection states
  - Implement state update logic that determines when re-rendering is needed
  - Add mouse position tracking and state change detection
  - Write helper functions to query current interaction state
  - _Requirements: 2.1, 2.2, 3.1, 3.4_

- [x] 5. Implement selective rendering engine





  - Create base renderer that displays clean A4 background without any shape overlays
  - Build preview renderer for hover state that shows outline-only shape highlighting
  - Implement selection renderer that displays full dimensions and measurements for selected shapes
  - Add dynamic instruction text rendering based on current interaction state
  - _Requirements: 1.1, 2.1, 3.1, 6.1, 6.2, 6.3, 6.4_

- [x] 6. Create mouse event handling system






  - Implement mouse move event handler that updates hover state and triggers re-render
  - Add mouse click event handler that updates selection state
  - Create console output for shape selection events
  - Add coordinate transformation in event handlers to work with original image space
  - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3_
-

- [x] 7. Integrate interactive system into main inspection workflow





  - Replace the existing static shape rendering in main.py with the new interactive system
  - Modify the shape detection pipeline to generate interactive shape data
  - Add proper initialization of interaction state and mouse callbacks
  - Ensure proper cleanup when exiting inspect mode
  - _Requirements: 1.1, 1.2, 7.1, 7.4_


- [x] 8. Add configuration parameters for interactive behavior






  - Add hover snapping distance configuration to config.py
  - Define preview and selection color constants
  - Create configurable thresholds for hit testing sensitivity
  - _Requirements: 2.4, 6.3, 6.4_



- [x] 9. Implement proper mode transition handling





  - Ensure clean state reset when returning to scan mode from inspect mode
  - Add proper camera capture re-initialization after inspect mode
  - Implement ESC key handling for application exit from inspect mode
  - Test mode transition robustness with proper resource cleanup
  - _Requirements: 7.1, 7.2, 7.3, 7.4_


- [x] 10. Create comprehensive test suite for interactive functionality




  - Write unit tests for hit testing accuracy with edge cases
  - Add integration tests for complete hover-to-click workflow
  - Create tests for coordinate transformation accuracy
  - Test selection priority logic with overlapping shapes
  - Add performance tests for real-time mouse interaction responsiveness
  - _Requirements: 4.1, 4.2, 4.3_


- [x] 11. Validate measurement accuracy preservation






  - Verify that all existing measurement calculations remain unchanged
  - Test A4 scaling factor consistency between old and new systems
  - Confirm dimension display matches original implementation exactly
  - Validate console output format compatibility
  - _Requirements: 5.1, 5.2, 5.3, 5.4_


- [x] 12. Optimize rendering performance for smooth interaction






  - Profile rendering performance during mouse movement
  - Implement efficient re-rendering that only updates when state changes
  - Add frame rate optimization for real-time mouse tracking
  - Test performance with multiple detected shapes
  - _Requirements: 2.1, 2.2_