# Requirements Document

## Introduction

Transform the existing OpenCV object detection script into an interactive "inspect mode" viewer that allows users to hover and click on detected shapes to view their dimensions. The system should provide a clean interface where shapes are hidden by default, with hover previews and click-to-inspect functionality similar to browser developer tools.

## Requirements

### Requirement 1

**User Story:** As a user, I want to see a clean A4 reference background without any detected shapes displayed by default, so that I can focus on individual objects when needed.

#### Acceptance Criteria

1. WHEN the inspect mode is activated THEN the system SHALL display only the warped A4 background image without any shape overlays
2. WHEN shapes are detected THEN the system SHALL store their contours and measurements internally without drawing them
3. WHEN no shapes are detected THEN the system SHALL display a "No valid object detected" message

### Requirement 2

**User Story:** As a user, I want to see a preview outline when I hover my mouse near a detected shape, so that I can identify which object I'm about to inspect.

#### Acceptance Criteria

1. WHEN the mouse cursor moves within a snapping distance of a detected shape THEN the system SHALL highlight the shape's outline in a preview color
2. WHEN the mouse cursor moves away from all shapes THEN the system SHALL remove all preview highlights
3. WHEN multiple shapes overlap at the cursor position THEN the system SHALL prioritize the smallest shape for more precise selection
4. WHEN the cursor is near a shape boundary THEN the system SHALL provide snapping behavior within a configurable distance threshold
5. WHEN a shape is already selected THEN the system SHALL NOT show hover preview for the selected shape

### Requirement 3

**User Story:** As a user, I want to click on a shape to select it and view its dimensions, so that I can inspect specific objects in detail.

#### Acceptance Criteria

1. WHEN I click on a detected shape THEN the system SHALL display only that shape with its full dimensions and measurements
2. WHEN I click on a shape THEN the system SHALL print the shape details to the console
3. WHEN I click on the background (no shape) THEN the system SHALL clear any current selection and print "None" to console
4. WHEN I click on a different shape THEN the system SHALL hide the previously selected shape and show the new selection
5. WHEN a shape is selected THEN the system SHALL display dimension arrows and measurement text as in the original implementation

### Requirement 4

**User Story:** As a user, I want accurate shape detection and hit testing, so that I can reliably select the intended objects without interference from shadows or noise.

#### Acceptance Criteria

1. WHEN performing hit testing THEN the system SHALL use the actual shape contours for accurate boundary detection
2. WHEN checking for mouse proximity THEN the system SHALL use cv2.pointPolygonTest for precise distance calculations
3. WHEN multiple shapes are at the same location THEN the system SHALL prioritize selection based on shape area (smallest first)
4. WHEN shapes have inner components (circles/rectangles) THEN the system SHALL treat them as separate selectable entities
5. WHEN filtering shapes THEN the system SHALL maintain the existing minimum area thresholds to avoid noise

### Requirement 5

**User Story:** As a user, I want to continue using A4 paper as the measurement reference, so that my dimension readings remain accurate and consistent.

#### Acceptance Criteria

1. WHEN measuring shapes THEN the system SHALL use the existing A4 scaling factors (mm_per_px_x, mm_per_px_y)
2. WHEN displaying dimensions THEN the system SHALL show measurements in millimeters as in the original system
3. WHEN warping the image THEN the system SHALL maintain the existing A4 detection and perspective correction
4. WHEN calculating measurements THEN the system SHALL preserve the existing PX_PER_MM scaling configuration

### Requirement 6

**User Story:** As a user, I want clear visual feedback about the current interaction state, so that I understand what actions are available.

#### Acceptance Criteria

1. WHEN no shape is hovered or selected THEN the system SHALL display "Hover to preview, click to inspect" instruction text
2. WHEN a shape is selected THEN the system SHALL display the shape type and dimensions in the instruction text
3. WHEN hovering over a shape THEN the system SHALL use a distinct preview color (different from selection color)
4. WHEN a shape is selected THEN the system SHALL use the original green color for consistency
5. WHEN displaying instruction text THEN the system SHALL position it clearly at the top of the image

### Requirement 7

**User Story:** As a user, I want to return to scanning mode after inspection, so that I can analyze different A4 sheets or re-scan the current one.

#### Acceptance Criteria

1. WHEN in inspect mode THEN the system SHALL wait for any key press to return to scanning mode
2. WHEN ESC is pressed in inspect mode THEN the system SHALL exit the application entirely
3. WHEN returning to scan mode THEN the system SHALL re-initialize the camera capture and reset stability counters
4. WHEN returning to scan mode THEN the system SHALL clear all previous detection state and start fresh
5. WHEN transitioning between modes THEN the system SHALL properly clean up OpenCV windows and resources