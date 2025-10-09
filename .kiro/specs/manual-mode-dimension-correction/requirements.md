# Requirements Document

## Introduction

The current A4 object dimension scanner has two detection modes: Auto Mode and Manual Mode. Auto Mode provides accurate dimension measurements but struggles with shape detection in complex scenarios. Manual Mode provides good shape detection through manual selection and shape snapping but produces incorrect dimension measurements. This feature will correct the dimension calculation logic in Manual Mode to match the accuracy of Auto Mode while preserving Manual Mode's superior shape detection capabilities.

## Requirements

### Requirement 1

**User Story:** As a user measuring objects with Manual Mode, I want the dimension measurements to be as accurate as Auto Mode, so that I can trust the measurement results regardless of which detection method I use.

#### Acceptance Criteria

1. WHEN Manual Mode detects a circle THEN the system SHALL calculate diameter measurements using the same mm_per_px scaling factors as Auto Mode
2. WHEN Manual Mode detects a rectangle THEN the system SHALL calculate width and height measurements using the same mm_per_px_x and mm_per_px_y scaling factors as Auto Mode
3. WHEN comparing measurements between Auto Mode and Manual Mode for the same object THEN the dimension values SHALL be within 2% tolerance
4. WHEN Manual Mode completes a measurement THEN the system SHALL use the A4 paper calibration scaling factors (mm_per_px_x, mm_per_px_y) for all dimension calculations
5. WHEN displaying Manual Mode results THEN the system SHALL show dimensions in millimeters with the same precision as Auto Mode

### Requirement 2

**User Story:** As a user switching between Auto Mode and Manual Mode, I want consistent measurement units and scaling, so that I can compare results directly without conversion.

#### Acceptance Criteria

1. WHEN the system processes manual selections THEN it SHALL receive and use the same mm_per_px_x and mm_per_px_y parameters as automatic detection
2. WHEN converting pixel measurements to millimeters THEN the system SHALL apply the scaling factors at the final measurement calculation step
3. WHEN the shape snapping engine detects shapes THEN it SHALL preserve pixel-based dimensions for accurate scaling conversion
4. WHEN manual measurement results are created THEN they SHALL use the same data structure and field names as automatic detection results
5. WHEN both modes measure the same object THEN they SHALL produce measurements that differ by less than the measurement precision threshold

### Requirement 3

**User Story:** As a user working with the measurement system, I want Manual Mode to maintain its current shape detection quality, so that I don't lose the ability to detect shapes that Auto Mode misses.

#### Acceptance Criteria

1. WHEN correcting dimension calculations THEN the system SHALL NOT modify the shape detection algorithms in the shape snapping engine
2. WHEN correcting dimension calculations THEN the system SHALL NOT modify the contour analysis or shape scoring logic
3. WHEN Manual Mode detects shapes THEN it SHALL continue to use the same shape snapping and selection area analysis
4. WHEN Manual Mode processes selections THEN it SHALL maintain the same user interaction workflow (click, drag, snap)
5. WHEN dimension corrections are applied THEN the system SHALL preserve all existing Manual Mode features including selection modes and visual feedback

### Requirement 4

**User Story:** As a developer maintaining the measurement system, I want the dimension correction to be implemented in the measurement conversion layer, so that changes are isolated and don't affect the core detection algorithms.

#### Acceptance Criteria

1. WHEN implementing dimension corrections THEN the system SHALL modify only the measurement conversion functions in measure.py
2. WHEN processing manual selections THEN the system SHALL pass mm_per_px scaling factors to the conversion functions
3. WHEN the shape snapping engine returns results THEN the conversion layer SHALL apply proper scaling to convert pixel dimensions to millimeters
4. WHEN integrating corrected measurements THEN the system SHALL use the existing measurement result data structure
5. WHEN the correction is complete THEN the system SHALL maintain backward compatibility with existing automatic detection workflows

### Requirement 5

**User Story:** As a user validating measurement accuracy, I want to be able to verify that Manual Mode and Auto Mode produce consistent results, so that I can have confidence in both measurement methods.

#### Acceptance Criteria

1. WHEN both modes measure the same circular object THEN the diameter measurements SHALL differ by less than 2mm or 2% whichever is larger
2. WHEN both modes measure the same rectangular object THEN the width and height measurements SHALL each differ by less than 2mm or 2% whichever is larger
3. WHEN measurement validation is performed THEN the system SHALL provide clear feedback about measurement consistency
4. WHEN dimension corrections are applied THEN the system SHALL maintain the same measurement precision (rounded to nearest millimeter)
5. WHEN displaying corrected measurements THEN the system SHALL use the same formatting and units as Auto Mode measurements